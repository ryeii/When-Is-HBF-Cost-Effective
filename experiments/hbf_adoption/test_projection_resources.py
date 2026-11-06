import unittest

from projection_resources import Events,Resource,Hop,chain,StreamedOperator


class CausalResources(unittest.TestCase):
    def test_later_hop_does_not_reserve_bandwidth_before_producer(self):
        events=Events();first=Resource(events,'first');second=Resource(events,'second',capture=True)
        completed=[]
        chain(events,[Hop(first,5),Hop(second,2)],lambda:completed.append(events.now))
        events.at(2,second.submit,3,lambda:completed.append(events.now))
        events.run()
        self.assertEqual(completed,[5,7])
        self.assertEqual(second.intervals,[dict(ready=2,start=2,finish=5),dict(ready=5,start=5,finish=7)])

    def test_pipeline_slots_remain_occupied_until_gpu_consumption(self):
        events=Events();source=Resource(events,'source');gpu=Resource(events,'gpu')
        completed=[]
        issued=[]
        def transfer(index):
            def reveal():
                issued.append((index,events.now))
                return [Hop(source,2)]
            return reveal
        operator=StreamedOperator(events,gpu,hot_gpu_seconds=3,
                    transfers=[transfer(i) for i in range(3)],cold_gpu_seconds=[4]*3,
                    staging_seconds=[1]*3,slots=2,callback=lambda:completed.append(events.now))
        events.run()
        self.assertEqual(operator.peak_outstanding,2)
        self.assertGreaterEqual(issued[2][1],8)
        self.assertEqual(len(completed),1)
        self.assertEqual(gpu.work,18)
        self.assertEqual(source.work,6)

    def test_malformed_or_reversed_events_fail(self):
        events=Events();events.at(1,lambda:None);events.run()
        with self.assertRaises(ValueError):events.at(0,lambda:None)
        with self.assertRaises(ValueError):Resource(events,'bad').submit(-1,lambda:None)

    def test_one_chunk_slot_serializes_consumption_and_next_issue(self):
        events=Events();source=Resource(events,'source');gpu=Resource(events,'gpu')
        completed=[]
        operator=StreamedOperator(events,gpu,hot_gpu_seconds=0,
                    transfers=[lambda:[Hop(source,2)]]*2,cold_gpu_seconds=[3,3],
                    staging_seconds=[1,1],slots=1,callback=lambda:completed.append(events.now))
        events.run()
        self.assertEqual(completed,[12])
        self.assertEqual(operator.peak_outstanding,1)


if __name__=='__main__':unittest.main()
