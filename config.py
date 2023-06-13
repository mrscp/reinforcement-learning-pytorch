from dotenv import load_dotenv
import os
load_dotenv()

storage_root = os.environ.get("STORAGE_ROOT")
model = os.environ.get("MODEL")
