from fast_api.access_models import AccessModels



class RuntimeState:
    """
    RuntimeState:
    A simple container object that holds the current state of the application. 
    - `api` stores an instance of AccessModels for performing model operations like embeddings, NER, or OCR. 
    - `uploaded_files` keeps track of files the user has uploaded during the session.
    Essentially, it centralizes shared objects and data so they can be accessed across different functions without passing them explicitly.
    """
    def __init__(self):
        self.api: AccessModels = None    

        self.uploaded_files = []
