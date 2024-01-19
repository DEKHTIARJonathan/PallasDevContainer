import os
import unittest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import jax
import jax.numpy as jnp
import jax_triton as jt  # Must be above `triton` import
from jax.experimental import pallas as pl

import triton
import triton.language as tl

import numpy as np


TEST_ARR_SIZE = 8


class TestPallasRunner(unittest.TestCase):

    def test_jax_install(self):

        @jax.jit
        def log2(x):
            ln_x = jnp.log(x)
            ln_2 = jnp.log(2.0)
            return ln_x / ln_2

        RNG = jax.random.PRNGKey(666)
        array_t = jax.random.uniform(RNG, (TEST_ARR_SIZE, 2), dtype=jnp.float32)

        # JIT the function first
        rslt = log2(array_t)
        self.assertEqual(rslt.shape, array_t.shape)

    def test_triton_install(self):
        @triton.jit
        def add_kernel(
            x_ptr,
            y_ptr,
            output_ptr,
            block_size: tl.constexpr,
        ):
            """Adds two vectors."""
            pid = tl.program_id(axis=0)
            block_start = pid * block_size
            offsets = block_start + tl.arange(0, block_size)
            mask = offsets < 8
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            output = x + y
            tl.store(output_ptr + offsets, output, mask=mask)
        
        def add(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            out_shape = jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)
            block_size = 8
            return jt.triton_call(
                x,
                y,
                kernel=add_kernel,
                out_shape=out_shape,
                grid=(x.size // block_size,),
                block_size=block_size)

        x_val = jnp.arange(TEST_ARR_SIZE)
        y_val = jnp.arange(TEST_ARR_SIZE, 16)

        rslt1 = add(x_val, y_val)
        rslt2 = jax.jit(add)(x_val, y_val)

        expected = np.array([8, 10, 12, 14, 16, 18, 20, 22])

        np.testing.assert_equal(expected, rslt1)
        np.testing.assert_equal(expected, rslt2)

    def test_jax_pallas(self):

        def add_vectors_kernel(x_ref, y_ref, o_ref):
            # x, y = x_ref[...], y_ref[...]
            # o_ref[...] = x + y
            o_ref[...] = x_ref[...] + y_ref[...]

        @jax.jit
        def add_vectors_pallas(x: jax.Array, y: jax.Array) -> jax.Array:
            return pl.pallas_call(
            add_vectors_kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            # debug=True
            )(x, y)

        x = jnp.arange(TEST_ARR_SIZE).astype(jnp.float32)
        y = jnp.arange(TEST_ARR_SIZE).astype(jnp.float32)

        rslt = add_vectors_pallas(x, y)
        expected = np.array(x + y)

        self.assertEqual(rslt.shape, (TEST_ARR_SIZE,))
        self.assertEqual(rslt.dtype, jnp.float32)
        np.testing.assert_equal(expected, rslt)


if __name__ == '__main__':
    unittest.main()
