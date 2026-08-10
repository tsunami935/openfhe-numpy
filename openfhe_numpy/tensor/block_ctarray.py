# ==================================================================================
#  BSD 2-Clause License
#
#  Copyright (c) 2014-2025, NJIT, Duality Technologies Inc. and other contributors
#
#  All rights reserved.
#
#  Author TPOC: contact@openfhe.org
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice, this
#     list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ==================================================================================

from openfhe import Ciphertext
from .block_tensor import BlockFHETensor
from .ctarray import CTArray
from ..utils.errors import ONP_ERROR
from ..openfhe_numpy import ArrayEncodingType

import numpy as np

class BlockCTArray(BlockFHETensor[Ciphertext]):
    tensor_priority = 40  # Higher priority than CTArray

    def __str__(self):
        return f"BlockCTArray(shape={self.original_shape}, cell={len(self._blocks)} {len(self._blocks[0]) if self._blocks else 0}, block_shape={self.block_shape})"

    def __repr__(self):
        return self.__str__()

    def clone(self, blocks=None):
        return BlockFHETensor(blocks.clone(), self.block_shape, self.original_shape, self.batch_size, self.ncols, self.order)

    def decrypt(self, secret_key, unpack_type="original"):
        stack = []
        for i in range(self.block_shape[0]):
            row = [block.decrypt(secret_key, unpack_type) for block in self.blocks[i * self.block_shape[1] : (i + 1) * self.block_shape[1]]]
            stack.append(np.concatenate(row, axis=1))
        return np.concatenate(stack, axis=0)[:self.original_shape[0], :self.original_shape[1]]

    def serialize(self) -> dict:
        """
        Serialize ciphertext and metadata to a dictionary.
        """
        block_data = [block.serialize() for block in self.blocks]

        data_dict =  {
            "original_shape": self.original_shape,
            "batch_size": self.batch_size,
            "block_shape": self.block_shape,
            "order": int(self.order),
            "ncols": self.ncols,
            "blocks": block_data,
        }
        return data_dict

    @classmethod
    def deserialize(cls, obj: dict) -> "BlockCTArray":
        """
        Deserialize a dictionary back into a CTArray.
        """
        required_keys = [
            "blocks",
            "original_shape",
            "block_shape",
            "batch_size",
            "ncols",
            "order",
        ]
        for key in required_keys:
            if key not in obj:
                ONP_ERROR(f"Missing required key '{key}' in serialized object.")

        blocks = [CTArray.deserialize(block_data) for block_data in obj['blocks']]
        return BlockCTArray(
            data = blocks,
            block_shape=obj["block_shape"],
            original_shape = obj["original_shape"],
            batch_size = obj["batch_size"],   
            ncols=obj["ncols"],         
            order = ArrayEncodingType.ROW_MAJOR if obj["order"] == 0 else ArrayEncodingType.COL_MAJOR,
        )
    
    def rescale(self, level=None):
        """Perform rescale."""
        for block in self.blocks:
            block.rescale(level)
        return self
    
    def bootstrap(self):
        """Perform bootstrap."""
        cc = self.blocks[0].data.GetCryptoContext()
        for i in range(len(self.blocks)):
            self.blocks[i].data = cc.EvalBootstrap(self.blocks[i].data)
        return self
    
    def retile_data(self):
        for block in self.blocks:
            block.retile_data()
        return self
