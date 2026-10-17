"""Component registry: the pipeline runs these in order (C1 -> C2 -> C3 -> C4)."""
from .c1 import C1Component
from .c2 import C2Component
from .c3 import C3Component
from .c4 import C4Component

C1 = C1Component()
PIPELINE = [C1, C2Component(), C3Component(), C4Component()]
BY_NAME = {c.name: c for c in PIPELINE}
