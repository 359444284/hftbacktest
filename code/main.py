import numba
import time

from numba import njit
import pandas as pd
import numpy as np

from numba.typed import Dict

from hftbacktest import NONE, NEW, HftBacktest, GTX, FeedLatency, SquareProbQueueModel, BUY, SELL, Linear, Stat, reset, \
    COL_PRICE
from hftbacktest.order import IOC, LIMIT, FOK, GTC, MARKET

@njit
def verify_order1(signal, av, bv):
    # return True
    if signal == 2:
        if (100 > av / bv > 0.95):
            return True
    elif signal == 1:
        if (100 > bv / av > 0.95):
            return True
    return False


@njit
def gridtrading(hbt, stat):
    max_position = 5
    order_qty = 0.001
    last_order_id = -1
    order_id = 0

    # Running interval in microseconds
    start_time = hbt.current_timestamp
    while hbt.elapse(500):
        # Clears cancelled, filled or expired orders.
        hbt.clear_inactive_orders()

        signal = hbt.get_user_data(110)[COL_PRICE]
        signal = 1 if signal == 2 else -1
        curr_ask = hbt.best_ask
        curr_bid = hbt.best_bid
        curr_ask_vol = hbt.ask_depth.get(hbt.best_ask_tick)
        curr_bid_vol = hbt.bid_depth.get(hbt.best_bid_tick)

        last_order_id = -1
        # Cancel all outstanding orders
        for order in hbt.orders.values():
            if order.cancellable:
                hbt.cancel(order.order_id)
                last_order_id = order.order_id

        # All order requests are considered to be requested at the same time.
        # Waits until one of the order cancellation responses is received.
        # Clears cancelled, filled or expired orders.
        if last_order_id >= 0:
            hbt.wait_order_response(last_order_id)
        # Clears cancelled, filled or expired orders.
        hbt.clear_inactive_orders()
        if hbt.position != 0 and (hbt.position * signal < 0):
            print(
                'current_timestamp:', (hbt.current_timestamp - start_time)/1000000,
                ', position:', hbt.position,
                ', equity:', hbt.equity, hbt.fee,
                'order_is: ', order_id,
                'signal: ', signal, curr_ask
            )
        if signal == np.nan:
            signal = 0

        if hbt.position == 0 and signal != 0:
            if verify_order1(signal, curr_ask_vol, curr_bid_vol):
                for order_id, order in hbt.orders.items():
                    # an order is only cancellable if order status is NEW.
                    # cancel request is negated if the order is already filled or filled before cancel request is processed.
                    if order.cancellable:
                        hbt.cancel(order_id)
                        # You can see status still NEW and see req CANCEL.
                        # cancels request also has order entry/response latencies the same as submitting.
                        hbt.wait_order_response(order_id)
                order_id += 1
                if signal == 1:
                    hbt.submit_buy_order(order_id,curr_ask,order_qty,GTX)
                    curr_state = 1
                    # print('buy1')
                elif signal == -1:
                    hbt.submit_sell_order(order_id,curr_bid,order_qty,GTX)
                    curr_state = -1
                    # print('sell1')
                else:
                    print('error 1', signal)
                last_order_id = order_id
        elif hbt.position != 0:
            if verify_order1(signal, curr_ask_vol, curr_bid_vol):
                for order_id, order in hbt.orders.items():
                    # an order is only cancellable if order status is NEW.
                    # cancel request is negated if the order is already filled or filled before cancel request is processed.
                    if order.cancellable:
                        hbt.cancel(order_id)
                        last_order_id = order_id
                        # You can see status still NEW and see req CANCEL.
                        # cancels request also has order entry/response latencies the same as submitting.
                        hbt.wait_order_response(order_id)

                if signal == 1 and hbt.position < 0.001:
                    order_id += 1
                    hbt.submit_buy_order(order_id,curr_ask,order_qty,GTX)
                    curr_state = 1
                    last_order_id = order_id
                elif signal == -1 and hbt.position > -0.001:
                    order_id += 1
                    hbt.submit_sell_order(order_id,curr_bid,order_qty,GTX)
                    last_order_id = order_id
                    curr_state = -1
                else:
                    pass


        # All order requests are considered to be requested at the same time.
        # Waits until one of the order responses is received.
        if last_order_id >= 0:
            if not hbt.wait_order_response(last_order_id):
                return False
        stat.record(hbt)

# @njit
# def gridtrading(hbt, stat):
#     max_position = 5
#     grid_interval = hbt.tick_size * 10
#     grid_num = 20
#     half_spread = hbt.tick_size * 20
#
#     # Running interval in microseconds
#     while hbt.elapse(100_000):
#         # Clears cancelled, filled or expired orders.
#         hbt.clear_inactive_orders()
#         print(hbt.current_timestamp, hbt.best_ask)
#
#         mid_price = (hbt.best_bid + hbt.best_ask) / 2.0
#         bid_order_begin = np.floor((mid_price - half_spread) / grid_interval) * grid_interval
#         ask_order_begin = np.ceil((mid_price + half_spread) / grid_interval) * grid_interval
#
#         order_qty = 0.1
#         last_order_id = -1
#
#         # Creates a new grid for buy orders.
#         new_bid_orders = Dict.empty(np.int64, np.float64)
#         if hbt.position < max_position:
#             for i in range(grid_num):
#                 bid_order_begin -= i * grid_interval
#                 bid_order_tick = round(bid_order_begin / hbt.tick_size)
#                 # Do not post buy orders above the best bid.
#                 if bid_order_tick > hbt.best_bid_tick:
#                     continue
#
#                 # order price in tick is used as order id.
#                 new_bid_orders[bid_order_tick] = bid_order_begin
#         for order in hbt.orders.values():
#             # Cancels if an order is not in the new grid.
#             if order.side == BUY and order.cancellable and order.order_id not in new_bid_orders:
#                 hbt.cancel(order.order_id)
#                 last_order_id = order.order_id
#         for order_id, order_price in new_bid_orders.items():
#             # Posts an order if it doesn't exist.
#             if order_id not in hbt.orders:
#                 hbt.submit_buy_order(order_id, order_price, order_qty, GTX)
#                 last_order_id = order_id
#
#         # Creates a new grid for sell orders.
#         new_ask_orders = Dict.empty(np.int64, np.float64)
#         if hbt.position > -max_position:
#             for i in range(grid_num):
#                 ask_order_begin += i * grid_interval
#                 ask_order_tick = round(ask_order_begin / hbt.tick_size)
#                 # Do not post sell orders below the best ask.
#                 if ask_order_tick < hbt.best_ask_tick:
#                     continue
#
#                 # order price in tick is used as order id.
#                 new_ask_orders[ask_order_tick] = ask_order_begin
#         for order in hbt.orders.values():
#             # Cancels if an order is not in the new grid.
#             if order.side == SELL and order.cancellable and order.order_id not in new_ask_orders:
#                 hbt.cancel(order.order_id)
#                 last_order_id = order.order_id
#         for order_id, order_price in new_ask_orders.items():
#             # Posts an order if it doesn't exist.
#             if order_id not in hbt.orders:
#                 hbt.submit_sell_order(order_id, order_price, order_qty, GTX)
#                 last_order_id = order_id
#
#         # All order requests are considered to be requested at the same time.
#         # Waits until one of the order responses is received.
#         if last_order_id >= 0:
#             if not hbt.wait_order_response(last_order_id):
#                 return False
#
#         # Records the current state for stat calculation.
#         stat.record(hbt)
#     return True

hbt = HftBacktest(
    [
        # '../hftbacktest/data/customer_data/btcusdt_20230613.npz',
        '../hftbacktest/data/customer_data/btcusdt_20230614.npz',
        '../hftbacktest/data/customer_data/btcusdt_20230615.npz',
        # '../hftbacktest/data/customer_data/btcusdt_20230616.npz',
    ],
    tick_size=0.1,
    lot_size=0.001,
    maker_fee=0,
    taker_fee=0.00017,
    order_latency=FeedLatency(),
    queue_model=SquareProbQueueModel(),
    asset_type=Linear,
    snapshot='../hftbacktest/data/modified_data/btcusdt_20230613_eod.npz'
)

stat = Stat(hbt)
print('start')
s = time.perf_counter()
gridtrading(hbt, stat.recorder)
e = time.perf_counter()
print(e-s)

stat.summary()
print('end')

