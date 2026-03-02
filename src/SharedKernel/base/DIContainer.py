import importlib
import inspect
import pkgutil
from lagom import Container as LagomContainer

class DIContainer(LagomContainer):
    def __init__(self, base_package: str = None):
        super().__init__()
        self.base_package = base_package or "src"
        self._registered_classes = set() 
        self._registered_interfaces = set()  
        self._scan_and_register()
    def _scan_and_register(self):
        print(f"[DI] Scanning package: {self.base_package}")
        package = importlib.import_module(self.base_package)
        for _, module_name, is_pkg in pkgutil.walk_packages(
            package.__path__, package.__name__ + "."
        ):
            print(f"[DI] Found module: {module_name} (is_pkg={is_pkg})")
            module = importlib.import_module(module_name)
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if hasattr(obj, "__di_type__"):
                    if obj in self._registered_classes:
                        print(f"[DI] ---- Skipping duplicate: {name}")
                        continue
                    
                    print(f"[DI] -> Candidate: {name}, type={obj.__di_type__}")
                    interfaces = [
                        base for base in obj.__bases__
                        if hasattr(base, "__name__") and base.__name__.startswith("I")
                    ]
                    if not interfaces:
                        print(f"[DI] ---- No Interface found for {name}")
                        continue
                    for interface in interfaces:
                        # Check if interface already registered using set
                        if interface in self._registered_interfaces:
                            # print(f"[DI] ---- Skipping duplicate interface: {interface.__name__}")
                            continue
                        
                        self[interface] = obj
                        self._registered_classes.add(obj)
                        self._registered_interfaces.add(interface)
                        print(
                            f"[DI] ---- Registered {interface.__name__} -> {name}"
                        )
        print("[DI] Scan completed!")