from ImplicitRDP.common.ring_buffer import RingBuffer


def test_peek_after_wraparound():
    buffer = RingBuffer(size=5)
    for i in range(10):
        buffer.push(i)

    items, error = buffer.peek_last_n(3)
    assert error is None
    assert items == [9, 8, 7]
