from dataclasses import dataclass, field
from typing import Optional, List
import pandas as pd
import numpy as np
import traceback
from pathlib import Path
import json
from typing import Optional, List, TextIO

@dataclass
class TradeState:
    cash: float
    btc: float
    peak_price: Optional[float] = None
    last_buy_time: Optional[pd.Timestamp] = None
    last_buy_index: Optional[int] = None
    last_trade_time: Optional[pd.Timestamp] = None
    trades: List[dict] = field(default_factory=list)
    trade_logs: List[dict] = field(default_factory=list)
    portfolio_values: List[float] = field(default_factory=list)
    traded_this_bar: bool = False
    preemptive_used_this_position: bool = False

    # logging bits:
    _log_fh: Optional[TextIO] = None
    _log_rec: Optional[dict] = None

    def open_log(self, run_dir: Path) -> Path:
        path = Path(run_dir) / "decision_log.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._log_fh = path.open("w", encoding="utf-8")
        return path

    def close_log(self):
        if self._log_fh:
            self._log_fh.close()
            self._log_fh = None

    def start_log_record(self, row, i):
        self._log_rec = {
            "time": str(row["parsed_time"]),
            "idx": int(i),
            "price": round(float(row["c"]), 6),
            "action": "hold",
            "reason": "none",
            "cash": float(self.cash),
            "btc": float(self.btc),
            "signal": int(row.get("Signal", 0)),
            "rsi": round(float(row.get("RSI", 0.0)), 2)
        }
                    
    def update_log_record(self, reason, action='hold'):
        self._log_rec["cash"] = float(self.cash)
        self._log_rec["btc"]  = float(self.btc)
        self._log_rec['reason'] = reason
        self._log_rec['action'] = action

    def write_log_record(self):
        if self._log_fh and self._log_rec is not None:
            rec = self._log_rec
            rec["cash"] = round(float(self.cash), 2)      # dollars/cents
            rec["btc"]  = round(float(self.btc), 6)       # BTC to 6 dp (adjust if you like)
            self._log_fh.write(json.dumps(self._log_rec, ensure_ascii=False) + "\n")
            self._log_rec = None  # clear for safety
    
def simulate_trading(df: pd.DataFrame, params, buy_model=None, sell_model=None, buy_features=None, sell_features=None, run_dir='.'):
    """
    One trade max per candle. Reasons are precise.
    Uses trailing stop (% from peak while holding) + optional max-hold (candles).
    """
    # Initial cash: keep your existing logic (uses 300th candle)
    initial_cash = df['c'].iloc[150]
    state = TradeState(cash=initial_cash, btc=0.0)
    
    trailing_stop_pct = float(params.get('trailing_stop_pct', 0.13))  # default ~13% like your earlier logic
    max_hold_candles = int(params.get('max_hold_time_candles', 1000))

    state.open_log(run_dir)

    for i, row in df.iterrows():
        price = row['c']
        current_time = row['parsed_time']
        # Portfolio mark
        state.portfolio_values.append(state.cash + state.btc * price)

        # Skip warmup period to match your earlier behavior
        if i < 150:
            continue

        # Update peak while holding
        if state.btc > 0:
            state.peak_price = price if state.peak_price is None else max(state.peak_price, price)
        else:
            state.peak_price = None  # reset when flat

        state.start_log_record(row, i)
        state.traded_this_bar = False

        state = _check_exit_conditions(
            row=row,
            i=i,
            state=state,
            trailing_stop_pct=trailing_stop_pct,
            max_hold_candles=max_hold_candles
        )
        if not state.traded_this_bar:
            state = _decide_buy(
                row=row,
                i=i,
                state=state,
                params=params,
                model=buy_model,
                model_features=buy_features
            )
                
        if not state.traded_this_bar:
            state = _decide_sell(
                row=row,
                i=i,
                state=state,
                params=params,
                model=sell_model,
                model_features=sell_features
            )
        state.write_log_record()

    state.close_log()
    print("Blocked soft sells:", getattr(state, "blocked_soft_sells", 0))
    vals = getattr(state, "confirm_pbears", [])
    if vals:
        q50,q80,q90,q95,q98 = np.quantile(vals, [0.5,0.8,0.9,0.95,0.98]).tolist()
        print(f"[CONFIRM DIST] med={q50:.3f} p80={q80:.3f} p90={q90:.3f} p95={q95:.3f} p98={q98:.3f}")
    return state.trades, state.portfolio_values, state.trade_logs, state.btc

def _ml_allows_buy(row, model, model_features, params) -> bool:
    """
    Return True if ML allows a buy.
    Respects params['ml_buy'] = { enabled, method: 'proba'|'label', threshold }.
    If disabled or no model/features, always allow (True).
    """
    cfg = (params or {}).get("ml_buy", {})

    # If ML filter disabled or model/feature list missing -> allow buy
    if not cfg.get("enabled", False) or model is None or model_features is None:
        return True

    try:
        X = row[model_features].values.reshape(1, -1)
        method = str(cfg.get("method", "proba")).lower()

        if method == "proba" and hasattr(model, "predict_proba"):
            thr = float(cfg.get("threshold", 0.75))
            p = float(model.predict_proba(X)[0][1])
            return p >= thr

        else:
            # fallback: hard label
            y = model.predict(X)[0]
            return int(y) == 1
    except Exception:
        # Fail-safe: if the model errors, block the buy (conservative).
        return False
    
def _check_exit_conditions(row, i, state, trailing_stop_pct, max_hold_candles):
    """Check trailing stop and max-hold (candles). Returns (state, traded_this_bar)."""
    if state.btc <= 0:
        return state

    price = row['c']
    now = row['parsed_time']

    # 1) Trailing stop: drop from peak by trailing_stop_pct
    if state.peak_price is not None and price < state.peak_price * (1 - trailing_stop_pct):
        # full exit on trailing stop
        state.cash += state.btc * price
        state.btc = 0.0
        state.trades.append({
            "type": "sell",
            "price": price,
            "cash": state.cash,        
            "btc": state.btc,        
            "parsed_time": now.isoformat(),
            "index": i,
            "reason": "trailing_stop"
        })
        state.peak_price = None
        state.last_buy_time = None
        state.last_buy_index = None
        state.last_trade_time = now
        state.traded_this_bar = True
        state.prev_p_bear = None
        state.preemptive_used_this_position = False
        state.update_log_record('trailing_stop', 'sell')
        return state

    # 2) Max-hold by candles
    if max_hold_candles is not None and state.last_buy_index is not None:
        if (i - state.last_buy_index) >= max_hold_candles:
            state.cash += state.btc * price
            state.btc = 0.0
            state.trades.append({
                "type": "sell",
                "price": price,
                "cash": state.cash,        
                "btc": state.btc,        
                "parsed_time": now.isoformat(),
                "index": i,
                "reason": "max_hold_candles"
            })
            state.peak_price = None
            state.last_buy_time = None
            state.last_buy_index = None
            state.last_trade_time = now
            state.traded_this_bar = True
            state.prev_p_bear = None
            state.preemptive_used_this_position = False
            state.update_log_record('max_hold_candles', 'sell')
            return state

    return state


def _decide_buy(row, i, state, params, model, model_features):
    """Rule + ML filter. Returns (state, traded_this_bar)."""
    if state.cash <= 0:
        return state

    signal = row['Signal']
    if not signal in [1, 2, 3]:
    #if not signal in [1, 2]:
        return state

    # --- NEW: regime-aware ML threshold (defaults keep existing behavior) ---
    bull = int(row.get('bull_regime', 0)) == 1  # requires you added this column; else defaults to 0
    ml_cfg = params.get("ml_buy", {}) or {}
    base_thr = float(ml_cfg.get("threshold", 0.5))
    # If these aren’t in config, they fall back to the current threshold
    bull_thr = float(ml_cfg.get("bull_threshold", base_thr))
    bear_thr = float(ml_cfg.get("bear_threshold", base_thr))
    eff_thr  = bull_thr if bull else bear_thr
    bypass_in_bull = bool(ml_cfg.get("bypass_in_bull", False))

    # Prepare an adjusted params to pass into your existing ML gate
    # (so we don't touch global params or change signatures)
    if bull and bypass_in_bull:
        ml_pass = True
    else:
        params_adj = dict(params)
        ml_adj = dict(ml_cfg)
        ml_adj["threshold"] = eff_thr
        params_adj["ml_buy"] = ml_adj
        ml_pass = _ml_allows_buy(row, model, model_features, params_adj)
    # --- END NEW ---
    if not ml_pass:
        state.update_log_record('no_ml_pass')
        return state
    
    # Size + reason
    if signal == 1:
        buy_fraction = 0.5
        reason = "buy_signal"
    elif signal == 2:
        buy_fraction = 1.0
        reason = "strong_buy_signal"
    elif signal == 3:
        buy_fraction = 0.15
        reason = "buy_adaptive_rsi"
    else:
        raise ValueError(f"Unexpected signal value: {signal}")

    price = row['c']
    now = row['parsed_time']
    buy_amount = state.cash * buy_fraction

    state.btc += buy_amount / price
    state.cash -= buy_amount
    state.trades.append({
        "type": "buy",
        "price": price,
        "cash": state.cash,        
        "btc": state.btc,        
        "parsed_time": now.isoformat(),
        "index": i,
        "reason": reason
    })
    state.last_buy_time = now
    state.last_buy_index = i
    state.last_trade_time = now
    state.traded_this_bar = True
    state.preemptive_used_this_position = False
    state.update_log_record(reason, 'buy')
    return state

def _ml_is_bearish(_row, model, features, ml_cfg) -> tuple[bool, float]:
    """(is_bearish, score). Fail-closed as (False,0.0)."""
    if not ml_cfg or not ml_cfg.get("enabled", False) or model is None or not features:
        return (False, 0.0)

    bull = int(_row.get('bull_regime', 0)) == 1
    base_thr = float(ml_cfg.get("preemptive_threshold", 0.75))
    bull_thr = float(ml_cfg.get("bull_threshold", base_thr)) if ml_cfg.get("bull_threshold") is not None else base_thr
    bear_thr = float(ml_cfg.get("bear_threshold", base_thr)) if ml_cfg.get("bear_threshold") is not None else base_thr
    thr = bull_thr if bull else bear_thr

    try:
        X = _row[features].to_numpy().reshape(1, -1)
        if ml_cfg.get("method", "proba") == "proba" and hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            # >>> FIX: pick the column that corresponds to class label 1 (bearish) <<<
            classes = getattr(model, "classes_", None)
            if classes is not None:
                try:
                    import numpy as np
                    bear_idx = int(np.where(classes == 1)[0][0])
                except Exception:
                    bear_idx = 1 if len(proba) > 1 else 0
            else:
                bear_idx = 1 if len(proba) > 1 else 0

            p_bear = float(proba[bear_idx])
            return (p_bear >= thr, p_bear)
        else:
            yhat = int(model.predict(X)[0])
            return (yhat == 1, float(yhat))

    except Exception as e:
        print(f"[ML-SELL ERROR] {type(e).__name__}: {e}")
        traceback.print_exc()
        return (False, 0.0)

def _ml_bear_proba(_row, model, features) -> float:
    """Return p(class==1) regardless of ml_cfg; 0.0 on failure."""
    try:
        # right before X = _row[features]...
        missing = [f for f in features if f not in _row.index]
        if missing:
            print("[ML] Missing features:", missing)
        elif _row[features].isna().any():
            print("[ML] NaNs in features at", _row.get("parsed_time"))

        X = _row[features].to_numpy().reshape(1, -1)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            classes = getattr(model, "classes_", None)
            if classes is not None:
                import numpy as np
                try:
                    idx = int(np.where(classes == 1)[0][0])
                except Exception:
                    idx = 1 if len(proba) > 1 else 0
            else:
                idx = 1 if len(proba) > 1 else 0
            return float(proba[idx])
        else:
            yhat = int(model.predict(X)[0])
            return float(yhat)  # 1 or 0
    except Exception as e:
        print(f"[ML-SELL CONFIRM ERROR] {type(e).__name__}: {e}")
        return 0.0
    

def _sell_fraction_from_signal(signal: int) -> float:
    return 1.0 if signal == -2 else 0.5

def _execute_sell(state: TradeState, price, now, i, signal):
    sell_fraction = _sell_fraction_from_signal(signal)
    sell_amount = state.btc * sell_fraction
    if sell_amount <= 0:
        return state

    state.btc -= sell_amount
    state.cash += sell_amount * price
    reason = "strong_sell_signal" if signal == -2 else "sell_signal"
    state.trades.append({
        "type": "sell",
        "price": float(price),
        "cash": state.cash,
        "btc": state.btc,
        "parsed_time": now.isoformat(),
        "index": int(i),
        "reason": reason
    })
    if state.btc <= 0:
        state.prev_p_bear = None
        state.preemptive_used_this_position = False
    state.last_trade_time = now
    state.traded_this_bar = True
    state.update_log_record(reason, 'sell')
    return state

def _soft_sell_passes_confirmation_fail_closed(row, state: TradeState, ml_cfg, model, model_features) -> bool:
    """Return True if soft sell (-1) passes confirmation; on error, BLOCK (fail-closed)."""
    confirm_thr = ml_cfg.get("confirm_rule_threshold", None)
    if confirm_thr is None or not ml_cfg.get("enabled", False) or model is None or not model_features:
        return True
    require_bull = ml_cfg.get("confirm_in_bull_only", False)
    if require_bull and int(row.get('bull_regime', 0)) != 1:
        return True
    try:
        p_bear = _ml_bear_proba(row, model, model_features)
        if not hasattr(state, "confirm_pbears"): state.confirm_pbears = []
        state.confirm_pbears.append(p_bear)
        return p_bear >= float(confirm_thr)
    except Exception:
        return False  # fail-closed

def _decide_sell(row, i, state, params, model, model_features):
    if state.btc <= 0:
        return state

    price = row['c']
    now = row['parsed_time']

    ml_cfg = params.get("ml_sell", {}) or {}

    if not state.preemptive_used_this_position:
        # --- ensure we have the tracking fields on state (no separate init needed) ---
        if not hasattr(state, "prev_p_bear"):
            state.prev_p_bear = None
        if not hasattr(state, "last_preemptive_index"):
            state.last_preemptive_index = None

        # --------------- A) Preemptive exit with crossing-only debounce ---------------
        is_bear, p_bear = _ml_is_bearish(row, model, model_features, ml_cfg)

        thr1 = float(ml_cfg.get("preemptive_threshold", 0.75))
        use_cross = bool(ml_cfg.get("use_crossing_only", True))

        # act only when p_bear crosses from below -> above thr1
        prev_pb = state.prev_p_bear
        crossing_ok = (prev_pb is None) or (prev_pb < thr1 <= p_bear) if use_cross else True

        # always update for next bar comparison
        state.prev_p_bear = p_bear

        # min-hold: don’t preempt right after a buy
        min_hold = int(ml_cfg.get("min_hold_bars_for_preemptive", 0))
        if state.last_buy_index is not None and (i - state.last_buy_index) < min_hold:
            is_bear = False

        # cooldown: don’t preempt again too soon
        cool = int(ml_cfg.get("cooldown_bars_after_preemptive", 0))
        last_idx = getattr(state, "last_preemptive_index", None)
        if is_bear and cool and last_idx is not None and (i - last_idx) < cool:
            is_bear = False

        # context guards
        if is_bear and ml_cfg.get("require_negative_momentum", False):
            if float(row.get("momentum", 0.0)) >= 0:
                is_bear = False

        if is_bear and ml_cfg.get("require_bear_regime", False):
            if int(row.get("bull_regime", 0)) == 1:
                is_bear = False
            
        if is_bear and p_bear >= thr1 and crossing_ok:
            frac = float(ml_cfg.get("fraction_on_preemptive", 1.0))
            frac = max(0.0, min(1.0, frac))
            if frac > 0.0:
                sell_amount = state.btc * frac
                state.btc -= sell_amount
                state.cash += sell_amount * price
                state.trades.append({
                    "type": "sell",
                    "price": float(price),
                    "cash": state.cash,        
                    "btc": state.btc,        
                    "parsed_time": now.isoformat(),
                    "index": int(i),
                    "reason": f"ml_preemptive_exit_p={p_bear:.3f}"
                })
                state.last_trade_time = now
                state.traded_this_bar = True
                state.preemptive_used_this_position = True             
                state.last_preemptive_index = i
                state.update_log_record('ml_preemptive_exit_p', 'sell')

                # optional: prevent a second (rule-based) sell on the same bar
                # if we don't return here, and we will do the second sell after this, the first trade is not logged
                if ml_cfg.get("block_rule_sell_same_bar_after_preemptive", True) or frac >= 0.999:
                    return state

    signal = row['Signal']
    if signal not in (-1, -2):
        return state

    confirm_thr = ml_cfg.get("confirm_rule_threshold", None)
    require_bull = ml_cfg.get("confirm_in_bull_only", False)
    if confirm_thr is not None and ml_cfg.get("enabled", False) and model is not None and model_features:
        if signal == -1:
            if not _soft_sell_passes_confirmation_fail_closed(row, state, ml_cfg, model, model_features):
                if not hasattr(state, "blocked_soft_sells"): state.blocked_soft_sells = 0
                state.blocked_soft_sells += 1
                state.update_log_record('blocked_soft_sells')
                return state

    state = _execute_sell(state, price, now, i, signal)
    return state


