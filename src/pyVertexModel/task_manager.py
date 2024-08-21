import asyncio
from multiprocessing import Pool

from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage

# Simula la ejecución de main.py
def run_main_py(process_id):
    print(f"Ejecutando instancia {process_id} de main.py")
    vModel = VertexModelVoronoiFromTimeImage()
    vModel.initialize()
    vModel.iterate_over_time()
    print(f"Instancia {process_id} de main.py terminada")
    return process_id

# Añade procesos asincrónicamente a la cola
async def async_add_task(pool, process_id):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, pool.apply, run_main_py, (process_id,))
    print(f"Resultado del proceso {process_id}: {result}")

# Procesa la cola de tareas
async def process_queue(queue, pool, max_processes):
    active_tasks = []

    while True:
        if len(active_tasks) < max_processes and not queue.empty():
            process_id = await queue.get()
            task = async_add_task(pool, process_id)
            active_tasks.append(task)

        if active_tasks:
            done, active_tasks = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)

        await asyncio.sleep(0.1)

# Inicializa el task manager y la cola
task_queue = asyncio.Queue()

async def main_task_manager():
    max_processes = 5
    with Pool(processes=max_processes) as pool:
        await process_queue(task_queue, pool, max_processes)

# Función para añadir tareas a la cola desde otro módulo
async def add_task(process_id):
    await task_queue.put(process_id)

# Arranca el task manager de forma asíncrona
async def start_task_manager():
    await main_task_manager()

if __name__ == "__main__":
    asyncio.run(start_task_manager())