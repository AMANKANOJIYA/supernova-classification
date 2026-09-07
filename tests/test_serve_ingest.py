import pytest

from src.schema import Observation, StaticContext, Survey
from src.serve.ingest import has_new_detection, parse_alert_packet
from src.serve.schemas import AlertPacket


def _static() -> StaticContext:
    return StaticContext(
        host_photoz=0.05, host_photoz_err=0.01, host_offset_arcsec=0.3,
        host_ellipticity=0.2, host_mag_r=18.5, mw_ebv=0.02, gal_lat=45.0,
        real_bogus_score=0.9,
    )


def _obs(mjd, flux=100.0, flux_err=5.0, is_detection=True) -> Observation:
    return Observation(mjd=mjd, lam_eff=472.3, flux=flux, flux_err=flux_err, is_detection=is_detection)


def _packet_bytes(observations, object_id="ZTF21abc", class_id=2222) -> bytes:
    packet = AlertPacket.from_alert_slice(
        object_id=object_id, survey=Survey.ZTF, observations=observations,
        static=_static(), class_id=class_id,
    )
    return packet.to_json()


class TestRoundTrip:
    def test_json_round_trip_preserves_fields(self):
        raw = _packet_bytes([_obs(60000.0, flux=-12.5)])
        normalized = parse_alert_packet(raw)
        assert normalized.object_id == "ZTF21abc"
        assert normalized.survey == Survey.ZTF
        assert normalized.class_id == 2222
        assert len(normalized.new_observations) == 1
        assert normalized.new_observations[0].flux == pytest.approx(-12.5)  # negative flux survives

    def test_negative_flux_and_none_class_id_survive(self):
        raw = _packet_bytes([_obs(60000.0, flux=-5.0)], class_id=None)
        normalized = parse_alert_packet(raw)
        assert normalized.class_id is None
        assert normalized.new_observations[0].flux < 0


class TestSorting:
    def test_out_of_order_observations_are_sorted(self):
        raw = _packet_bytes([_obs(60010.0), _obs(60000.0), _obs(60005.0)])
        normalized = parse_alert_packet(raw)
        mjds = [o.mjd for o in normalized.new_observations]
        assert mjds == [60000.0, 60005.0, 60010.0]


class TestValidation:
    def test_empty_observations_raises(self):
        raw = _packet_bytes([])
        with pytest.raises(ValueError, match="empty observations"):
            parse_alert_packet(raw)

    def test_non_positive_flux_err_raises(self):
        raw = _packet_bytes([_obs(60000.0, flux_err=0.0)])
        with pytest.raises(ValueError, match="flux_err"):
            parse_alert_packet(raw)


class TestHasNewDetection:
    def test_true_when_any_row_is_a_detection(self):
        raw = _packet_bytes([_obs(60000.0, is_detection=False), _obs(60001.0, is_detection=True)])
        assert has_new_detection(parse_alert_packet(raw)) is True

    def test_false_when_every_row_is_forced_photometry(self):
        raw = _packet_bytes([_obs(60000.0, is_detection=False)])
        assert has_new_detection(parse_alert_packet(raw)) is False
