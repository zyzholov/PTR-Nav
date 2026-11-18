"""Read the leading Header stamp for bounded raw-message diagnostic caches.

SE2Grid begins with SE2GridInfo.header; the other observed messages begin with
Header directly. Only ordinary CDR1 is accepted. This is never navigation code.
"""
import struct


def leading_stamp_ns(payload):
    if len(payload)<12:raise ValueError('short serialized Header')
    tag=int.from_bytes(payload[:2],'big')
    if tag not in (0,1):raise ValueError('diagnostic supports CDR1 only')
    sec,nsec=struct.unpack_from('<iI' if tag else '>iI',payload,4)
    if sec<0 or nsec>=1_000_000_000:raise ValueError('invalid diagnostic source stamp')
    return sec*1_000_000_000+nsec
