"""Example compact telemetry payload; not an installed radio protocol.

The abstract simulator uses a 128-byte beacon. This prototype reserves 24
bytes for a future MAC and fits measurements/intent/cache into the remainder.
Actual PHY/MAC/encryption overhead must be checked for the selected hardware.
"""
import binascii
import math
import struct

HEADER=struct.Struct('<BBBBII3f3f2fIhI')
PEER=struct.Struct('<BhH')
FRAME_BYTES=128
RESERVED_MAC_BYTES=24


def uint(value,bits,name):
    if isinstance(value,bool) or not isinstance(value,int) or not 0<=value<2**bits:raise ValueError('Invalid '+name)
    return value


def pack(sender,sequence,transmit_ms,position,velocity,goal,goal_issued_ms,energy_dbm=None,energy_sample_ms=None,peers=()):
    uint(sender,8,'sender');uint(sequence,32,'sequence');uint(transmit_ms,32,'transmission time');uint(goal_issued_ms,32,'goal time')
    if sender>=10 or goal_issued_ms>transmit_ms:raise ValueError('Invalid sender or future goal')
    if len(position)!=3 or len(velocity)!=3 or len(goal)!=2 or not all(math.isfinite(x) for x in (*position,*velocity,*goal)):
        raise ValueError('Finite pose and issued goal required')
    if len(peers)>9 or len({p[0] for p in peers})!=len(peers):raise ValueError('Invalid peer cache')
    flags=0;energy=0;sample=0
    if energy_dbm is not None:
        uint(energy_sample_ms,32,'energy measurement time')
        if not math.isfinite(energy_dbm) or not -200<=energy_dbm<=100 or energy_sample_ms>transmit_ms:
            raise ValueError('Invalid/future energy measurement')
        flags=1;energy=round(energy_dbm*10);sample=energy_sample_ms
    payload=HEADER.pack(1,sender,len(peers),flags,sequence,transmit_ms,*position,*velocity,*goal,goal_issued_ms,energy,sample)
    for peer,sinr,age_ms in peers:
        uint(peer,8,'peer');uint(age_ms,16,'peer measurement age')
        if peer>=10 or peer==sender or not math.isfinite(sinr) or not -100<=sinr<=100:raise ValueError('Invalid peer measurement')
        payload+=PEER.pack(peer,round(sinr*100),age_ms)
    payload+=struct.pack('<H',binascii.crc_hqx(payload,0xffff))
    if len(payload)+RESERVED_MAC_BYTES>FRAME_BYTES:raise ValueError('Telemetry exceeds the simulated beacon budget')
    return payload


def unpack(payload):
    if len(payload)<HEADER.size+2 or len(payload)+RESERVED_MAC_BYTES>FRAME_BYTES:raise ValueError('Invalid packet length')
    values=HEADER.unpack_from(payload);version,sender,count,flags,sequence,transmit_ms=values[:6]
    expected=HEADER.size+count*PEER.size+2
    if version!=1 or sender>=10 or count>9 or flags not in (0,1) or len(payload)!=expected:raise ValueError('Invalid packet header')
    if struct.unpack_from('<H',payload,len(payload)-2)[0]!=binascii.crc_hqx(payload[:-2],0xffff):raise ValueError('Corrupt telemetry packet')
    position=values[6:9];velocity=values[9:12];goal=values[12:14];issued,energy,sample=values[14:17]
    if issued>transmit_ms or (flags and sample>transmit_ms):raise ValueError('Packet claims future observations')
    if not all(math.isfinite(x) for x in (*position,*velocity,*goal)):raise ValueError('Nonfinite telemetry')
    peers=[]
    for k in range(count):
        peer,sinr,age=PEER.unpack_from(payload,HEADER.size+k*PEER.size)
        if peer>=10 or peer==sender or any(p[0]==peer for p in peers):raise ValueError('Invalid cached peer')
        peers.append((peer,sinr/100,age))
    return dict(sender=sender,sequence=sequence,transmit_ms=transmit_ms,position=position,velocity=velocity,
                goal=goal,goal_issued_ms=issued,energy_dbm=energy/10 if flags else None,
                energy_sample_ms=sample if flags else None,peers=peers)
