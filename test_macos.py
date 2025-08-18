import platform

def is_macos():
    """Check if the current operating system is macOS."""
    return platform.system() == 'Darwin'

# Usage
if is_macos():
    print("This device is running macOS")
else:
    print("This device is not running macOS")