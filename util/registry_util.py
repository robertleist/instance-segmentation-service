# registry_utils.py
_PENDING_REGISTRATIONS = []

def register_base_model():
    """Decorator to mark a class for automatic registration."""
    def decorator(cls):
        # We store the class and the metadata for later
        _PENDING_REGISTRATIONS.append(cls)
        return cls
    return decorator