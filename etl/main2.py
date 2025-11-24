import json
import pyodbc
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import List, Tuple

# Configure logging - only to console, minimal file logging with rotation
from logging.handlers import RotatingFileHandler

# Create rotating file handler (max 10MB, keep 2 backup files)
file_handler = RotatingFileHandler(
    'match_details_extraction.txt',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=2
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        file_handler,
        logging.StreamHandler()
    ]
)

# Thread-safe counter
class Counter:
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()
    
    def increment(self, amount=1):
        with self.lock:
            self.value += amount
            return self.value

# Excluded league IDs
EXCLUDED_LEAGUES = {336, 121, 9837, 9986, 10282, 9441, 9296, 8972, 130}

def get_connection(connection_string: str):
    """Create a new database connection for thread safety."""
    return pyodbc.connect(connection_string)

def process_batch(connection_string: str, batch: List[Tuple], counter: Counter, total: int, batch_num: int):
    """Process a batch of matches in a single thread."""
    conn = get_connection(connection_string)
    cursor = conn.cursor()
    batch_start = datetime.now()
    
    try:
        from matchDetails import process_single_match
        
        successful = 0
        failed = 0
        
        for row in batch:
            source_id, match_id, league_id, league_name, season_year, raw_json = row
            try:
                # Process the match
                success = process_single_match(conn, match_id, raw_json)
                
                if success:
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                logging.error(f"Error processing match {match_id}: {str(e)}")
                failed += 1
        
        # Bulk update extraction status for the entire batch
        try:
            # Get all match_ids from this batch
            match_ids = [row[1] for row in batch]  # row[1] is match_id
            
            # Use table-valued parameter approach with temp table for better performance
            cursor.execute("CREATE TABLE #batch_matches (match_id INT)")
            
            # Insert batch match IDs into temp table
            for match_id in match_ids:
                cursor.execute("INSERT INTO #batch_matches VALUES (?)", match_id)
            
            # Merge update
            cursor.execute("""
                MERGE INTO [dbo].[extraction_status] AS target
                USING #batch_matches AS source
                ON target.match_id = source.match_id
                WHEN MATCHED THEN
                    UPDATE SET has_match_details = 1, extraction_time = GETDATE()
                WHEN NOT MATCHED THEN
                    INSERT (match_id, has_match_details, extraction_time)
                    VALUES (source.match_id, 1, GETDATE());
            """)
            
            cursor.execute("DROP TABLE #batch_matches")
            conn.commit()
            
        except Exception as e:
            logging.error(f"Error in batch update: {str(e)}")
            conn.rollback()
        
        # Update progress - only log every 500 matches to reduce file I/O
        count = counter.increment(len(batch))
        batch_duration = (datetime.now() - batch_start).total_seconds()
        batch_speed = len(batch) / batch_duration if batch_duration > 0 else 0
        
        # Only log every 10th batch to reduce disk writes
        if batch_num % 10 == 0:
            logging.info(f"Batch {batch_num}: {len(batch)} matches in {batch_duration:.1f}s ({batch_speed:.1f}/s) | Total: {count}/{total} ({count/total*100:.1f}%) | Success: {successful}, Failed: {failed}")
                
    finally:
        conn.close()

def get_total_count(connection_string: str, excluded_str: str) -> int:
    """Get total count without fetching all data."""
    conn = get_connection(connection_string)
    cursor = conn.cursor()
    
    cursor.execute(f"""
        SELECT COUNT(*)
        FROM [dbo].[fotmob_raw_data] WITH (NOLOCK)
        WHERE data_type = 'match_details' 
        AND league_id NOT IN ({excluded_str})
    """)
    
    total = cursor.fetchone()[0]
    conn.close()
    return total

def fetch_batch_of_matches(connection_string: str, excluded_str: str, offset: int, fetch_size: int) -> List[Tuple]:
    """Fetch a batch of matches using OFFSET-FETCH."""
    conn = get_connection(connection_string)
    cursor = conn.cursor()
    
    query = f"""
        SELECT f.id, f.match_id, f.league_id, f.league_name, f.season_year, f.raw_json
        FROM [dbo].[fotmob_raw_data] f WITH (NOLOCK)
        WHERE f.data_type = 'match_details' 
        AND f.league_id NOT IN ({excluded_str})
        ORDER BY f.match_id
        OFFSET ? ROWS
        FETCH NEXT ? ROWS ONLY
    """
    
    cursor.execute(query, offset, fetch_size)
    matches = cursor.fetchall()
    conn.close()
    return matches

def main(connection_string: str, max_workers: int = 6, process_batch_size: int = 50, fetch_batch_size: int = 5000):
    """
    Main extraction function with chunked processing.
    
    Args:
        max_workers: Number of parallel threads
        process_batch_size: Matches per processing batch (for parallelization)
        fetch_batch_size: Number of matches to fetch from DB at once (to avoid loading all into memory)
    """
    
    logging.info("Starting fast match details extraction...")
    
    # Build exclusion list
    excluded_str = ','.join(str(x) for x in EXCLUDED_LEAGUES)
    
    # Get total count first
    logging.info("Counting total matches...")
    total_matches = get_total_count(connection_string, excluded_str)
    logging.info(f"Total matches to process: {total_matches:,} (excluded {len(EXCLUDED_LEAGUES)} leagues)")
    
    if total_matches == 0:
        logging.info("No matches to process")
        return
    
    # Process in chunks to avoid loading everything into memory
    counter = Counter()
    start_time = datetime.now()
    offset = 0
    overall_batch_num = 0
    
    while offset < total_matches:
        # Fetch a chunk of matches
        chunk_start = datetime.now()
        logging.info(f"\n--- Fetching matches {offset:,} to {min(offset + fetch_batch_size, total_matches):,} ---")
        matches = fetch_batch_of_matches(connection_string, excluded_str, offset, fetch_batch_size)
        
        if not matches:
            break
        
        fetch_time = (datetime.now() - chunk_start).total_seconds()
        logging.info(f"Fetched {len(matches)} matches in {fetch_time:.1f}s")
        
        # Create processing batches from this chunk
        batches = []
        for i in range(0, len(matches), process_batch_size):
            batches.append(matches[i:i + process_batch_size])
        
        logging.info(f"Processing {len(batches)} batches with {max_workers} workers...")
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for batch in batches:
                overall_batch_num += 1
                futures.append(
                    executor.submit(process_batch, connection_string, batch, counter, total_matches, overall_batch_num)
                )
            
            # Wait for all to complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Batch processing error: {str(e)}")
        
        # Move to next chunk
        offset += fetch_batch_size
        
        # Show progress summary
        elapsed = (datetime.now() - start_time).total_seconds()
        avg_speed = counter.value / elapsed if elapsed > 0 else 0
        remaining = total_matches - counter.value
        eta_seconds = remaining / avg_speed if avg_speed > 0 else 0
        eta_minutes = eta_seconds / 60
        
        logging.info(f"\n=== Chunk Complete === Progress: {counter.value:,}/{total_matches:,} ({counter.value/total_matches*100:.1f}%) | Speed: {avg_speed:.1f}/s | ETA: {eta_minutes:.1f} min\n")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Final summary
    logging.info("\n" + "="*60)
    logging.info("=== EXTRACTION COMPLETE ===")
    logging.info("="*60)
    logging.info(f"Total matches processed: {counter.value:,}")
    logging.info(f"Time taken: {duration/60:.1f} minutes ({duration:.1f} seconds)")
    logging.info(f"Average speed: {counter.value/duration:.1f} matches/second")
    logging.info("="*60)

if __name__ == "__main__":
    # Connection string
    connection_string = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=DESKTOP-J9IV3OH;"
        "DATABASE=fussballDB;"
        "Trusted_Connection=yes;"
    )
    
    # Optimized parameters:
    # max_workers: 6-8 threads for parallel processing
    # process_batch_size: 50 matches per thread batch
    # fetch_batch_size: 5000 matches fetched from DB at once (controls memory usage)
    main(
        connection_string, 
        max_workers=8,           # Parallel threads
        process_batch_size=50,   # Matches per processing batch
        fetch_batch_size=5000    # Fetch 5000 at a time to manage memory
    )