process runPythonScript {
    queue 'myqueue'
    script:
    """
    source util/add_to_pythonpath.sh
    python main.py
    """
}

workflow {
    runPythonScript
}