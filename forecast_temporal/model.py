"""All pretrained graph parameters remain trainable; temporal depth is explicit."""
from forecast_graph.model import ForecastGraph

class TemporalGraph(ForecastGraph):
    def __init__(self,variant):
        super().__init__('hybrid_shared');self.temporal_variant=variant

    def forward(self,node,edge,**kw):
        if self.temporal_variant.startswith('short_'):node=node[:,-4:];edge=edge[:,-4:]
        return super().forward(node,edge,**kw)
