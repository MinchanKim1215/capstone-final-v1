"""Own process instrumentation: exact original draws, separate truth and sensors."""
from contextlib import contextmanager
import numpy as np
from .instrumented_packet import traced_packet_interval
from swarm_sim.fast_network import graph_statistics

class Recorder:
    def __init__(self,env):
        self.env=env;p=env.packets;size=len(p.source);n=env.n;ns=env.ns
        self.entry=np.zeros(size);self.first=np.full(size,-1.);self.queue_wait=np.zeros(size);self.service_wait=np.zeros(size);self.end_time=np.full(size,-1.)
        self.queue_before=np.zeros(size);self.service_before=np.zeros(size)
        self.evidence=np.zeros((size,5),np.int32);self.last_next=np.full((n,ns),-1,np.int32);self.last_next_time=np.full((n,ns),-1.)
        self.frame_extra=np.zeros((n,4),np.int32);self.extra_history=[];self.observed=[];self.physical=[];self.nojam=np.zeros((n,n));self.blocked=np.zeros((n,n),bool)

    def sensor(self,now):
        p=self.env.packets;n=self.env.n;ns=self.env.ns;out=np.zeros((n,27),np.float32)
        sums=self.frame_extra+(self.extra_history[-1] if self.extra_history else 0)
        for node in range(n):
            ids=p.queue[node,:p.lengths[node]];age=now-p.born[ids]
            if len(ids):
                out[node,0]=age.max();out[node,1]=age.mean();out[node,2]=np.count_nonzero(age>.25+1e-10);out[node,3]=np.count_nonzero((age>=.20)&(age<=.25+1e-10));out[node,4]=np.count_nonzero(p.tries[ids]>0)
                out[node,8:8+ns]=np.bincount(p.dest[ids],minlength=ns)
            out[node,5:8]=sums[node,:3];out[node,14:20]=-1;out[node,20:26]=np.inf
            out[node,14:14+ns]=self.last_next[node];out[node,20:20+ns]=np.where(self.last_next_time[node]>=0,now-self.last_next_time[node],np.inf);out[node,26]=sums[node,3]
        return out

    @contextmanager
    def installed(self):
        import cooperative.packets as packets
        import adaptive_joint_expanded.world as world
        old_kernel=packets.packet_interval;old_hop=world.hop_probabilities
        def hop(loss,interference,shadow,received,sinr):
            actual=old_hop(loss,interference,shadow,received,sinr);env=self.env;n=env.n;q=env.t%6
            a=env.city.points[env.node_log[-2]];b=env.city.points[env.node_log[-1]];xy=a*(1-q/6)+b*(q/6);xyz=np.c_[xy,np.full(n,50.)]
            geom,los,extra=env.city.radio.propagation(np.repeat(xyz,n,axis=0),np.tile(xyz,(n,1)),env.scenario)
            np.testing.assert_allclose(geom.reshape(n,n),loss,rtol=0,atol=1e-11)
            self.blocked=~los.reshape(n,n);scratch=np.empty((n,n));scratch2=np.empty((n,n))
            self.nojam=old_hop(loss,np.zeros(n),shadow,scratch,scratch2)
            nobuild=old_hop(loss-extra.reshape(n,n),interference,shadow,scratch,scratch2)
            nofade=old_hop(loss,interference,np.zeros_like(shadow),scratch,scratch2)
            conn=[float(graph_statistics(p,env.ns)[1][0]) for p in (actual,self.nojam,nobuild,nofade)]
            self.physical.append(np.r_[conn,float(np.any(interference)),float(self.blocked.mean())].astype(np.float32))
            return actual
        def kernel(*args):
            result=traced_packet_interval(*args,self.entry,self.first,self.queue_wait,self.service_wait,self.queue_before,self.service_before,self.end_time,self.evidence,self.nojam,self.blocked,self.last_next,self.last_next_time,self.frame_extra)
            self.observed.append(self.sensor((int(args[5])+1)*float(args[4])));self.extra_history.append(self.frame_extra.copy());return result
        packets.packet_interval=kernel;world.hop_probabilities=hop
        try:yield self
        finally:packets.packet_interval=old_kernel;world.hop_probabilities=old_hop

    def arrays(self):
        p=self.env.packets;g=int(p.ic[0]);out=dict(observed=np.asarray(self.observed),physical_truth=np.asarray(self.physical),
            packet_queue_wait=self.queue_wait[:g],packet_service_wait=self.service_wait[:g],packet_queue_before_deadline=self.queue_before[:g],packet_service_before_deadline=self.service_before[:g],packet_end=self.end_time[:g],packet_evidence_before_deadline=self.evidence[:g],
            packet_born=p.born[:g],packet_status=p.status[:g],packet_delay=p.delays[:g],packet_source=p.source[:g],packet_dest=p.dest[:g],
            jam_start=np.array(self.env.jam_start),jam_end=np.array(self.env.jam_end))
        # Last2s draining is deliberately not used for event attribution. Eligible
        # event windows end8.5s before missionend, so all contributing packets
        # have settled in the instrumented live interval sequence.
        good=(out['packet_status']==1)&(out['packet_end']>=0)
        error=np.abs(out['packet_queue_wait'][good]+out['packet_service_wait'][good]-out['packet_delay'][good])
        if error.size and error.max()>1e-8:raise ValueError('Per-hop delay components do not conserve end-to-end delay')
        out['delay_decomposition_max_error']=np.array(error.max() if error.size else 0.)
        settled=out['packet_end']>=0
        expected=np.clip(out['packet_end'][settled]-out['packet_born'][settled],0,.25)
        err=np.abs(out['packet_queue_before_deadline'][settled]+out['packet_service_before_deadline'][settled]-expected)
        if err.size and err.max()>1e-8:raise ValueError('Pre-deadline wait components do not conserve elapsed time')
        out['predeadline_decomposition_max_error']=np.array(err.max() if err.size else 0.)
        return out
