import asyncio

import numpy as np

from src.pyVertexModel import task_manager

async def main():
    """
    Add new routines to the task manager with random IDs
    :return:
    """
    random_id = np.random.randint(0, 10000)
    await task_manager.add_task(random_id)

async def run_all():
    await asyncio.gather(
        task_manager.start_task_manager(),
        main()
    )

# Ejecuta el código asíncrono
if __name__ == "__main__":
    asyncio.run(run_all())