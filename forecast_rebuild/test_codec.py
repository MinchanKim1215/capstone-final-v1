import unittest
from .beacon_codec import pack,unpack,RESERVED_MAC_BYTES


class CodecTests(unittest.TestCase):
    def data(self):
        return dict(sender=3,sequence=45,transmit_ms=1500,position=(10.,20.,50.),velocity=(1.,0.,0.),
                    goal=(30.,40.),goal_issued_ms=0,energy_dbm=-94.1,energy_sample_ms=1000,
                    peers=[(j,12.34,1000) for j in range(10) if j!=3])

    def test_maximum_fleet_fits_original_beacon_budget(self):
        data=self.data();packet=pack(**data);restored=unpack(packet)
        self.assertLessEqual(len(packet)+RESERVED_MAC_BYTES,128)
        self.assertEqual(restored['energy_dbm'],-94.1)
        self.assertEqual(restored['energy_sample_ms'],1000)
        self.assertEqual(len(restored['peers']),9)

    def test_missing_energy_is_none_instead_of_zero(self):
        data=self.data();data.update(energy_dbm=None,energy_sample_ms=None)
        self.assertIsNone(unpack(pack(**data))['energy_dbm'])

    def test_corruption_and_future_observation_rejected(self):
        data=self.data();packet=bytearray(pack(**data));packet[10]^=1
        with self.assertRaises(ValueError):unpack(bytes(packet))
        with self.assertRaises(ValueError):pack(**dict(data,energy_sample_ms=1501))


if __name__=='__main__':unittest.main()
