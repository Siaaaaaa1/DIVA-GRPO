import os
import argparse
import pandas as pd
import multiprocessing
import time
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, Any, List, Optional
import pyarrow as pa
import pyarrow.parquet as pq

# Import local module
from llm_client import AzureQwenClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def writer_listener(queue: multiprocessing.Queue, output_path: str):
    """
    Listener process that consumes results from the queue and writes to a single Parquet file incrementally.
    """
    buffer = []
    writer = None
    BATCH_SIZE = 100
    total_written = 0

    logger.info(f"Writer listener started. Output file: {output_path}")

    while True:
        # Get record from queue
        record = queue.get()
        
        # Check for kill signal
        if record == "KILL":
            break
            
        buffer.append(record)
        
        # Flush buffer if full
        if len(buffer) >= BATCH_SIZE:
            writer = flush_buffer(buffer, output_path, writer)
            total_written += len(buffer)
            buffer = []
            
    # Final flush
    if buffer:
        writer = flush_buffer(buffer, output_path, writer)
        total_written += len(buffer)

    if writer:
        writer.close()
        
    logger.info(f"Writer finished. Total rows written: {total_written}")

def flush_buffer(buffer: List[Dict], output_path: str, writer: Optional[pq.ParquetWriter]) -> pq.ParquetWriter:
    """Helper to write a buffer to the parquet file."""
    try:
        df_batch = pd.DataFrame(buffer)
        table = pa.Table.from_pandas(df_batch)
        
        # Initialize writer on first batch
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema)
        
        # Handle schema mismatch if necessary (e.g. alignment)
        if table.schema != writer.schema:
            table = table.cast(writer.schema)
            
        writer.write_table(table)
    except Exception as e:
        logger.error(f"Failed to write batch: {e}")
        
    return writer

def process_chunk_worker(chunk_df: pd.DataFrame, queue: multiprocessing.Queue, api_config: Dict[str, str], worker_id: int):
    """
    Worker process: Generates data and puts it into the queue.
    """
    # Initialize client inside the child process
    client = AzureQwenClient(
        api_key=api_config['key'],
        endpoint=api_config['endpoint'],
        deployment=api_config.get('deployment', 'qwen-vl-max-latest')
    )
    
    # Use tqdm position to avoid overlapping bars
    for idx, row in tqdm(chunk_df.iterrows(), total=len(chunk_df), desc=f"Worker-{worker_id}", position=worker_id):
        try:
            problem = row['problem']
            answer = row['answer']
            row_id = row.get('id', idx)
            
            # Extract image bytes
            image_data = None
            if 'images' in row and row['images']:
                img_entry = row['images']
                if isinstance(img_entry, dict) and 'bytes' in img_entry:
                    image_data = img_entry['bytes']
                elif isinstance(img_entry, bytes):
                    image_data = img_entry
            
            # 1. Generate Variants
            variants = client.generate_variants(problem)
            
            # 2. Generate Think Steps (Verification included)
            think_result = client.generate_think_steps(problem, answer, image_data)
            
            if think_result['status'] == 'success':
                item = row.to_dict()
                item['variants'] = variants
                item['think_steps'] = think_result['think_steps']
                item['think_answer'] = think_result['think_answer']
                
                # Send to writer
                queue.put(item)
            else:
                # Optional: Log failure reason slightly less frequently to avoid spam
                pass
                
        except Exception as e:
            logger.error(f"Worker-{worker_id} error on row {row.get('id')}: {e}")
            continue

def main():
    parser = argparse.ArgumentParser(description="Augment dataset with Variants and Think Steps using LLM.")
    parser.add_argument("--input", type=str, required=True, help="Path to input Parquet file")
    parser.add_argument("--output", type=str, required=True, help="Path to output Parquet file (Single merged file)")
    parser.add_argument("--workers", type=int, default=4, help="Number of multiprocessing workers")
    parser.add_argument("--api_key", type=str, default=os.getenv("AZURE_OPENAI_KEY"), help="Azure API Key")
    parser.add_argument("--endpoint", type=str, default=os.getenv("AZURE_OPENAI_ENDPOINT"), help="Azure Endpoint")
    
    args = parser.parse_args()
    
    if not args.api_key or not args.endpoint:
        raise ValueError("API Key and Endpoint must be provided.")

    api_config = {
        "key": args.api_key,
        "endpoint": args.endpoint,
        "deployment": "qwen-vl-max-latest"
    }

    # Load Dataset
    logger.info(f"Loading dataset from {args.input}...")
    df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df)} rows.")
    
    # Prepare Output Dir
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    # Use Manager Queue for IPC
    manager = multiprocessing.Manager()
    queue = manager.Queue()

    # Start Writer Listener
    listener = multiprocessing.Process(target=writer_listener, args=(queue, str(output_path)))
    listener.start()

    # Start Workers
    chunk_size = (len(df) + args.workers - 1) // args.workers
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    workers = []
    logger.info(f"Starting {len(chunks)} workers...")
    
    for i, chunk in enumerate(chunks):
        p = multiprocessing.Process(
            target=process_chunk_worker,
            args=(chunk, queue, api_config, i)
        )
        workers.append(p)
        p.start()

    # Wait for all workers to finish
    for p in workers:
        p.join()

    # Signal listener to stop
    queue.put("KILL")
    listener.join()

    logger.info("Processing complete. All data saved to single file.")

if __name__ == "__main__":
    main()