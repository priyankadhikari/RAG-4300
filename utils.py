import time
import psutil
import os
import csv

def measure_time_memory(func, *args, **kwargs):
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss  # in bytes
    start_time = time.time()

    result = func(*args, **kwargs)

    end_time = time.time()
    mem_after = process.memory_info().rss  # in bytes

    exec_time = end_time - start_time
    mem_used = (mem_after - mem_before) / (1024 * 1024)  # Convert to MB

    return exec_time, mem_used, result

def write_to_csv(data, filename="experiment_results.csv"):
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            # Write header if file does not exist.
            writer.writerow([
                "VectorDB", "LLM", "EmbeddingModel", "ChunkSize", "Overlap",
                "IngestTime(s)", "IngestMem(MB)",
                "SearchTime(s)", "SearchMem(MB)", "Query", "Response"
            ])
        writer.writerow(data)
