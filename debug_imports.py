
try:
    import app_utils
    print(f"File: {app_utils.__file__}")
    print(f"Attributes: {dir(app_utils)}")
    from app_utils import apply_theme
    print("SUCCESS: apply_theme imported")
except ImportError as e:
    print(f"ERROR: {e}")
except Exception as e:
    print(f"EXCEPTION: {e}")
