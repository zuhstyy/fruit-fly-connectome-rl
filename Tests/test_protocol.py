from contextlib import contextmanager
import json
from pathlib import Path
import socket
import sys
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python"))
from protocol import (
    Client,
    BridgeError,
    ControlPaused,
    ControlStopped,
    PROTOCOL,
    VERSION,
)


@contextmanager
def server(handler):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)

    def run():
        c, _ = sock.accept()
        with c, c.makefile("rwb") as f:

            def send(data):
                f.write((json.dumps(data) + "\n").encode())
                f.flush()

            send(dict(type="hello", protocol=PROTOCOL, version=VERSION))
            while line := f.readline():
                q = json.loads(line)
                send(handler(q))
                if q["type"] == "release":
                    break

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        yield sock.getsockname()[1]
    finally:
        sock.close()
        thread.join(timeout=2)


class ProtocolTests(unittest.TestCase):
    def test_feedback_is_sent_immediately_without_another_action(self):
        requests = []

        def reply(q):
            requests.append(q)
            return dict(type="feedback", id=q["id"])

        with server(reply) as port:
            c = Client(port, "secret")
            c.set_feedback(15, -8, -0.1, -5, "BACKTRACK")
            self.assertEqual(len(requests), 1)
            self.assertEqual(requests[0]["type"], "feedback")
            self.assertEqual(requests[0]["reward_score"], -5)
            self.assertEqual(requests[0]["reward_last"], -8)
            c.close(False)

    def test_shutdown_sends_authenticated_quit(self):
        requests = []

        def reply(q):
            requests.append(q)
            return dict(type="quitting", id=q["id"])

        with server(reply) as port:
            c = Client(port, "secret")
            c.close(quit_game=True)
        self.assertEqual(requests[0]["type"], "quit")
        self.assertEqual(requests[0]["token"], "secret")

    def test_pause_and_stop_have_distinct_control_errors(self):
        for code, error in [
            ("control_paused", ControlPaused),
            ("control_stopped", ControlStopped),
        ]:
            with server(lambda q: dict(type="error", id=q["id"], code=code)) as port:
                c = Client(port, "secret")
                with self.assertRaises(error):
                    c.step([1, 1, 0, 0, 0, 0], 4)
                c.close(False)

    def test_auth_and_request_correlation(self):
        requests = []

        def reply(q):
            requests.append(q)
            return dict(type="status", id=q["id"])

        with server(reply) as port:
            c = Client(port, "secret")
            c.request("status")
            c.close()
        self.assertEqual([q["id"] for q in requests], [1, 2])
        self.assertTrue(all(q["token"] == "secret" for q in requests))

    def test_error_and_wrong_id_are_not_success(self):
        for response in (
            dict(type="error", id=1, message="bad state"),
            dict(type="state", id=999),
        ):
            with server(lambda q: response) as port:
                c = Client(port, "secret")
                with self.assertRaises(BridgeError):
                    c.request("status")
                c.close(False)


if __name__ == "__main__":
    unittest.main()
