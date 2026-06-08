
Quick Start
=============

.. code-block:: python

   import numpy as np
   from openfhe import *
   import openfhe_numpy as onp
   
   # Initialize CKKS context
   params = CCParamsCKKSRNS()
   params.SetMultiplicativeDepth(7)
   params.SetScalingModSize(59)
   params.SetFirstModSize(60)
   params.SetScalingTechnique(FIXEDAUTO)
   params.SetSecretKeyDist(UNIFORM_TERNARY)
   
   cc = GenCryptoContext(params)
   cc.Enable(PKESchemeFeature.PKE)
   cc.Enable(PKESchemeFeature.LEVELEDSHE)
   cc.Enable(PKESchemeFeature.ADVANCEDSHE)
   
   # Generate keys
   keys = cc.KeyGen()
   cc.EvalMultKeyGen(keys.secretKey)
   cc.EvalSumKeyGen(keys.secretKey)
   
   # Create matrix and encrypt it
   A = np.array([[1, 2], [3, 4]])
   
   ring_dim = cc.GetRingDimension()
   batch_size = ring_dim // 2
   
   # Encrypt with OpenFHE-NumPy
   ctm_A = onp.array(
         cc=cc,
         data=A,
         batch_size=batch_size,
         order=onp.ROW_MAJOR,
         fhe_type="C",
         mode="tile",
         public_key=keys.publicKey,
      )
   
   
   # Generate keys
   onp.EvalSquareMatMultRotateKeyGen(keys.secretKey, ctm_A.ncols)
   
   # Perform encrypted operations
   ctm_product = ctm_A @ ctm_A      # Matrix multiplication
   ctm_sum = onp.add(ctm_A, ctm_A)  # Element-wise addition
   
   # Decrypt results
   decrypted_product = ctm_product.decrypt(keys.secretKey, unpack_type="original")
   decrypted_sum = ctm_sum.decrypt(keys.secretKey, unpack_type="original")
   
   print("Result of A @ A:")
   print(decrypted_product)
   print("Expected result:")
   print(A @ A)
   
   print("Result of A + A:")
   print(decrypted_sum)
   print("Expected result:")
   print(A + A)
