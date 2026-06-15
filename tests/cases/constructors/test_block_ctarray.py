import gc
import numpy as np
from openfhe import *
import openfhe_numpy as onp
from core import *

from openfhe_numpy.utils import is_power_of_two

SIZES = [2, 3, 8, 16]
ORDERS = [("row_major", onp.ROW_MAJOR)]

def _ensure_depth(params: dict, min_depth: int = 3) -> dict:
    p = params.copy()
    if params.get("multiplicativeDepth", 0) < min_depth:
        p["multiplicativeDepth"] = min_depth
    return p

class TestBlockConstructor(MainUnittest):
    def test_block_constructor_square(self):
        ckks_params = load_ckks_params()

        for p in ckks_params:
            params = _ensure_depth(p, 3)
            batch_size = params["ringDim"] // 2
            cc, keys = gen_crypto_context(params)
            cc.EvalMultKeyGen(keys.secretKey)
            cc.EvalSumKeyGen(keys.secretKey)

            try:
                for size in SIZES:
                    if params["ringDim"] < 4096 and size > 2:
                        continue
                    if size > batch_size:
                        continue
                    if not is_power_of_two(int(np.sqrt(batch_size))):
                        continue

                    A = generate_random_array(rows=size, cols=size)

                    for order_name, order_value in ORDERS:
                        with self.subTest(order=order_name, size=size, ringDim=params["ringDim"]):
                            result = None
                            ctm = None
                            try:
                                ctm = onp.block_array(
                                    cc=cc,
                                    data=A,
                                    batch_size=batch_size,
                                    order=order_value,
                                    fhe_type="C",
                                    mode="tile",
                                    public_key=keys.publicKey,
                                )
                                result = ctm.decrypt(keys.secretKey, unpack_type="original")
                                self.assertArrayClose(actual=result, expected=A)
                            except Exception:
                                self._record_case(
                                    params={
                                        "case": "total_sum",
                                        "size": size,
                                        "ringDim": p["ringDim"],
                                    },
                                    input_data={"A": A},
                                    expected=A,
                                    result=result,
                                )
                                raise
                            finally:
                                del ctm, result
                                gc.collect()
            finally:
                del cc, keys
                gc.collect()