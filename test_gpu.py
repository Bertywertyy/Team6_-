import tensorflow as tf
import sys

print("=" * 60)
print("🚀 GPU ACCELERATION TEST")
print("=" * 60)

# Check TensorFlow version
print(f"\n✅ TensorFlow version: {tf.__version__}")

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
print(f"\n🔍 GPUs detected: {len(gpus)}")

if gpus:
    for i, gpu in enumerate(gpus):
        print(f"   GPU {i}: {gpu.name}")
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"   ✓ Memory growth enabled for GPU {i}")
        except RuntimeError as e:
            print(f"   ⚠️ {e}")
    print("\n✅ GPU ACCELERATION IS ENABLED!")
else:
    print("\n❌ NO GPU DETECTED - Running on CPU")
    print("   This is normal if you don't have CUDA drivers installed")

# Quick performance test
print("\n🧪 Running quick matrix multiplication test...")
try:
    with tf.device('/GPU:0' if gpus else '/CPU:0'):
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)
        print(f"   ✓ Test passed on {'GPU' if gpus else 'CPU'}")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 60)
print("Test complete! Your app will use GPU if available.")
print("=" * 60)
