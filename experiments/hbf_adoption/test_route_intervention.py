"""Actual expert unions and training-only identities in the route intervention."""
import unittest
import numpy as np
from route_intervention import RouteSimulator


class RouteInterventionTests(unittest.TestCase):
    def test_current_batch_union_and_assignment_fraction(self):
        sim=object.__new__(RouteSimulator)
        # Every held-out token routes to 0/1; training ranking caches expert 0.
        sim.data=np.zeros((1,8000,2),dtype=np.int16);sim.data[:,:,1]=1
        sim.ranking=np.array([[0,1]])
        sim.layer_plan={0:(0.,0.,1,32.)}
        sim.requests={'a':dict(session=0,processed=9,input_tokens=16,token_times_s=[]),
                      'b':dict(session=1,processed=2,input_tokens=16,token_times_s=[])}
        sim.active_batch=dict(requests=['a','b'],queries=[3,1],prefill=True)
        size,fraction=sim._route(0,4)
        self.assertEqual(size,32.)  # One unique cold expert across four tokens.
        self.assertEqual(fraction,.5)
        sim.layer_plan[0]=(0.,0.,2,32.)
        self.assertEqual(sim._route(0,4),(0.,0.))

    def test_correlated_routes_disable_meanfield_bounds(self):
        sim=object.__new__(RouteSimulator)
        sim.layout=type('Layout',(),dict(replicas=2))()
        sim.requests={'a':dict(predecessor=None,output_tokens=64)}
        bound=sim.service_bounds()
        self.assertEqual(bound['decode_interval_s'],0)
        self.assertEqual(bound['mandatory_resource_work_s'],{})
        self.assertEqual(bound['passing_roots_upper_bound'],1)


if __name__=='__main__':unittest.main()
