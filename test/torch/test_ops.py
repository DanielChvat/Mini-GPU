import gc
import os
import unittest

try:
    import torch
    import torch_mini_gpu
except ImportError as exc:
    raise unittest.SkipTest(f"PyTorch Mini-GPU tests require torch: {exc}") from exc


PORT = os.environ.get("MINIGPU_TEST_PORT")
BAUD = int(os.environ.get("MINIGPU_TEST_BAUD", "115200"))
MEMORY_SIZE = int(os.environ.get("MINIGPU_TEST_MEMORY_SIZE", "65536"))

if os.environ.get("MINIGPU_TEST_DEBUG", "").lower() not in {"1", "true", "yes", "on"}:
    os.environ.pop("MINIGPU_COMM_DEBUG", None)
    os.environ.pop("MINIGPU_RUNTIME_LOG", None)


@unittest.skipUnless(PORT, "set MINIGPU_TEST_PORT=/dev/ttyUSB* to run hardware tests")
class MiniGpuTorchOpsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch_mini_gpu.connect(PORT, baud=BAUD, memory_size=MEMORY_SIZE)

    @classmethod
    def tearDownClass(cls):
        torch_mini_gpu.disconnect()

    def tearDown(self):
        gc.collect()

    def to_minigpu(self, tensor):
        return tensor.contiguous().to("minigpu")

    def assert_tensor_close(self, actual, expected, *, rtol=1e-3, atol=1e-3):
        actual_cpu = actual.to("cpu")
        self.assertEqual(actual_cpu.shape, expected.shape)
        self.assertEqual(actual_cpu.dtype, expected.dtype)
        torch.testing.assert_close(actual_cpu, expected, rtol=rtol, atol=atol)

    def test_cpu_to_minigpu_and_back(self):
        x = torch.tensor([-3, -1, 0, 2, 4], dtype=torch.int8)
        self.assert_tensor_close(self.to_minigpu(x), x, rtol=0, atol=0)

    def test_dtype_conversions_float_half_int(self):
        x = torch.tensor([-2.25, -0.5, 0.0, 1.5, 3.75], dtype=torch.float32)
        gx = self.to_minigpu(x)

        self.assert_tensor_close(gx.float(), x.float())
        self.assert_tensor_close(gx.half(), x.half(), rtol=1e-3, atol=1e-3)
        self.assert_tensor_close(gx.int(), x.int(), rtol=0, atol=0)
        self.assert_tensor_close(gx.short(), x.short(), rtol=0, atol=0)
        self.assert_tensor_close(gx.char(), x.char(), rtol=0, atol=0)

    @unittest.skipUnless(hasattr(torch, "float8_e4m3fn"), "PyTorch build lacks float8_e4m3fn")
    def test_dtype_conversion_float8_e4m3fn(self):
        x = torch.tensor([-1.0, -0.25, 0.0, 0.5, 1.0], dtype=torch.float32)
        expected = x.to(torch.float8_e4m3fn)
        self.assert_tensor_close(self.to_minigpu(x).to(torch.float8_e4m3fn), expected, rtol=0, atol=0)

    def test_add_mul_div(self):
        a = torch.tensor(
            [-8.0, -3.5, -1.0, -0.125, 0.0, 0.25, 1.0, 3.5, 8.0],
            dtype=torch.float32,
        )
        b = torch.tensor(
            [-4.0, -1.75, -0.5, 0.25, 0.5, 1.0, 2.0, -0.25, 0.125],
            dtype=torch.float32,
        )
        ga = self.to_minigpu(a)
        gb = self.to_minigpu(b)

        self.assert_tensor_close(ga + gb, a + b)
        self.assert_tensor_close(ga * gb, a * b)
        self.assert_tensor_close(ga / gb, a / b, rtol=2e-2, atol=2e-2)

    def test_vector_add_integer_dtypes(self):
        for dtype in (torch.int32, torch.int16, torch.int8):
            with self.subTest(dtype=dtype):
                a = torch.tensor([1, -2, 3, -4], dtype=dtype)
                b = torch.tensor([5, 6, -7, -8], dtype=dtype)
                self.assert_tensor_close(self.to_minigpu(a) + self.to_minigpu(b), a + b, rtol=0, atol=0)

    def test_relu(self):
        x = torch.tensor([-3.0, -0.5, 0.0, 2.0, 4.0], dtype=torch.float32)
        self.assert_tensor_close(torch.relu(self.to_minigpu(x)), torch.relu(x))

    def test_sqrt_and_reciprocal(self):
        x = torch.tensor(
            [0.001, 0.01, 0.0625, 0.25, 0.5, 1.0, 2.0, 4.0, 9.0, 16.0, 64.0, 256.0],
            dtype=torch.float32,
        )
        gx = self.to_minigpu(x)

        self.assert_tensor_close(torch.sqrt(gx), torch.sqrt(x), rtol=2e-2, atol=2e-2)
        self.assert_tensor_close(torch.reciprocal(gx), torch.reciprocal(x), rtol=2e-2, atol=2e-2)

    def test_exp_sigmoid_tanh(self):
        x = torch.tensor([-10.0, -5.0, -2.0, -1.5, -1.0, -0.25, 0.0, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0], dtype=torch.float32)
        gx = self.to_minigpu(x)

        self.assert_tensor_close(torch.exp(gx), torch.exp(x), rtol=5e-2, atol=1e-4)
        self.assert_tensor_close(torch.sigmoid(gx), torch.sigmoid(x), rtol=5e-2, atol=5e-2)
        self.assert_tensor_close(torch.tanh(gx), torch.tanh(x), rtol=5e-2, atol=5e-2)

    def test_log_log2_pow(self):
        x = torch.tensor([0.001, 0.01, 0.1, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 256.0], dtype=torch.float32)
        y = torch.tensor([0.5, -0.25, 2.0, 3.0, 2.0, 1.0, 2.0, 0.5, 3.0, -1.0, 0.25, 0.125], dtype=torch.float32)
        gx = self.to_minigpu(x)
        gy = self.to_minigpu(y)

        self.assert_tensor_close(torch.log(gx), torch.log(x), rtol=1e-3, atol=1e-3)
        self.assert_tensor_close(torch.log2(gx), torch.log2(x), rtol=1e-3, atol=1e-3)
        self.assert_tensor_close(torch.pow(gx, gy), torch.pow(x, y), rtol=5e-2, atol=5e-2)
        self.assert_tensor_close(torch.pow(gx, 2.0), torch.pow(x, 2.0), rtol=1e-3, atol=1e-3)

    def test_log10_and_trig(self):
        pi = torch.pi
        x = torch.tensor([0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0], dtype=torch.float32)
        angles = torch.tensor(
            [
                -4.0 * pi,
                -3.5 * pi,
                -3.0 * pi,
                -2.5 * pi,
                -2.0 * pi,
                -1.5 * pi,
                -pi,
                -0.75 * pi,
                -0.5 * pi,
                -0.25 * pi,
                -1.0,
                -0.5,
                0.0,
                0.5,
                1.0,
                0.25 * pi,
                0.5 * pi,
                0.75 * pi,
                pi,
                1.5 * pi,
                2.0 * pi,
                2.5 * pi,
                3.0 * pi,
                3.5 * pi,
                4.0 * pi,
            ],
            dtype=torch.float32,
        )
        gx = self.to_minigpu(x)
        ga = self.to_minigpu(angles)

        self.assert_tensor_close(torch.log10(gx), torch.log10(x), rtol=1e-4, atol=1e-4)
        self.assert_tensor_close(torch.sin(ga), torch.sin(angles), rtol=5e-2, atol=5e-2)
        self.assert_tensor_close(torch.cos(ga), torch.cos(angles), rtol=5e-2, atol=5e-2)
        self.assert_tensor_close(torch.tan(ga), torch.tan(angles), rtol=1e-1, atol=1e-1)

    def test_reductions_argmax_argmin_softmax(self):
        x = torch.tensor([1.5, -2.0, 4.0, 0.25, -3.0, 2.0], dtype=torch.float32)
        gx = self.to_minigpu(x)

        self.assert_tensor_close(torch.sum(gx), torch.sum(x))
        self.assert_tensor_close(torch.mean(gx), torch.mean(x), rtol=2e-2, atol=2e-2)
        self.assert_tensor_close(torch.amax(gx), torch.amax(x))
        self.assert_tensor_close(torch.amin(gx), torch.amin(x))
        self.assert_tensor_close(torch.argmax(gx), torch.argmax(x).to(torch.int32), rtol=0, atol=0)
        self.assert_tensor_close(torch.argmin(gx), torch.argmin(x).to(torch.int32), rtol=0, atol=0)
        self.assert_tensor_close(torch.ops.minigpu.sum(gx), torch.sum(x))
        self.assert_tensor_close(torch.ops.minigpu.mean(gx), torch.mean(x), rtol=2e-2, atol=2e-2)
        self.assert_tensor_close(torch.ops.minigpu.amax(gx), torch.amax(x))
        self.assert_tensor_close(torch.ops.minigpu.amin(gx), torch.amin(x))
        self.assert_tensor_close(torch.ops.minigpu.argmax(gx), torch.argmax(x).to(torch.int32), rtol=0, atol=0)
        self.assert_tensor_close(torch.ops.minigpu.argmin(gx), torch.argmin(x).to(torch.int32), rtol=0, atol=0)

        logits = torch.tensor([[1.0, 2.0, -1.0], [0.5, -0.5, 3.0]], dtype=torch.float32)
        self.assert_tensor_close(
            torch.nn.functional.softmax(self.to_minigpu(logits), dim=-1),
            torch.nn.functional.softmax(logits, dim=-1),
            rtol=5e-2,
            atol=5e-2,
        )
        self.assert_tensor_close(
            torch.ops.minigpu.softmax(self.to_minigpu(logits), -1),
            torch.nn.functional.softmax(logits, dim=-1),
            rtol=5e-2,
            atol=5e-2,
        )

    def test_pool2d_kernels(self):
        x = torch.tensor(
            [[[[1.0, -2.0, 0.5, 3.0],
               [0.25, 1.5, -0.5, 2.0],
               [2.5, -1.0, 4.0, 0.75],
               [-0.25, 3.5, 1.0, -1.5]]]],
            dtype=torch.float32,
        )
        gx = self.to_minigpu(x)

        self.assert_tensor_close(
            torch.nn.functional.max_pool2d(gx, kernel_size=2, stride=2),
            torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2),
        )
        self.assert_tensor_close(
            torch.ops.minigpu.max_pool2d(gx, [2, 2], [2, 2]),
            torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2),
        )
        self.assert_tensor_close(
            torch.nn.functional.avg_pool2d(gx, kernel_size=2, stride=2),
            torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2),
            rtol=2e-2,
            atol=2e-2,
        )
        self.assert_tensor_close(
            torch.ops.minigpu.avg_pool2d(gx, [2, 2], [2, 2]),
            torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2),
            rtol=2e-2,
            atol=2e-2,
        )
        self.assert_tensor_close(
            torch.nn.functional.adaptive_avg_pool2d(gx, output_size=(2, 2)),
            torch.nn.functional.adaptive_avg_pool2d(x, output_size=(2, 2)),
            rtol=2e-2,
            atol=2e-2,
        )
        self.assert_tensor_close(
            torch.ops.minigpu.adaptive_avg_pool2d(gx, [2, 2]),
            torch.nn.functional.adaptive_avg_pool2d(x, output_size=(2, 2)),
            rtol=2e-2,
            atol=2e-2,
        )

    def test_mm_and_linear(self):
        a = torch.tensor([[1.0, 2.0, -1.0], [0.5, -2.0, 4.0]], dtype=torch.float32)
        b = torch.tensor([[2.0, -1.0], [0.0, 3.0], [1.5, 0.5]], dtype=torch.float32)
        weight = torch.tensor([[1.0, 0.5, -1.0], [2.0, -0.25, 0.75]], dtype=torch.float32)

        self.assert_tensor_close(torch.mm(self.to_minigpu(a), self.to_minigpu(b)), torch.mm(a, b))
        self.assert_tensor_close(
            torch.nn.functional.linear(self.to_minigpu(a), self.to_minigpu(weight)),
            torch.nn.functional.linear(a, weight),
        )

    def test_blas_dot_mv_addmm(self):
        x = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float32)
        y = torch.tensor([-0.5, 3.0, 4.0], dtype=torch.float32)
        a = torch.tensor([[1.0, 2.0, -1.0], [0.5, -2.0, 4.0]], dtype=torch.float32)
        b = torch.tensor([[2.0, -1.0], [0.0, 3.0], [1.5, 0.5]], dtype=torch.float32)
        bias = torch.tensor([[0.25, -0.5], [1.0, 0.75]], dtype=torch.float32)

        self.assert_tensor_close(torch.dot(self.to_minigpu(x), self.to_minigpu(y)), torch.dot(x, y))
        self.assert_tensor_close(torch.mv(self.to_minigpu(a), self.to_minigpu(x)), torch.mv(a, x))
        self.assert_tensor_close(
            torch.addmm(self.to_minigpu(bias), self.to_minigpu(a), self.to_minigpu(b), beta=0.5, alpha=1.5),
            torch.addmm(bias, a, b, beta=0.5, alpha=1.5),
        )
        self.assert_tensor_close(torch.ops.minigpu.scal(self.to_minigpu(x), 2.5), x * 2.5)
        self.assert_tensor_close(torch.ops.minigpu.axpy(self.to_minigpu(x), self.to_minigpu(y), -0.25), -0.25 * x + y)

    def test_tiny_mnist_mlp_forward(self):
        model_cpu = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(28 * 28, 4, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 10, bias=True),
        ).eval()

        with torch.no_grad():
            model_cpu[1].weight.copy_(
                torch.linspace(-0.05, 0.05, 4 * 28 * 28, dtype=torch.float32).view(4, 28 * 28)
            )
            model_cpu[1].bias.copy_(torch.linspace(-0.2, 0.2, 4, dtype=torch.float32))
            model_cpu[3].weight.copy_(
                torch.linspace(-0.2, 0.2, 10 * 4, dtype=torch.float32).view(10, 4)
            )
            model_cpu[3].bias.copy_(torch.linspace(0.1, -0.1, 10, dtype=torch.float32))

        image = torch.linspace(-1.0, 1.0, 28 * 28, dtype=torch.float32).view(1, 1, 28, 28)
        model_gpu = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(28 * 28, 4, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 10, bias=True),
        ).eval()
        model_gpu.load_state_dict(model_cpu.state_dict())
        model_gpu = model_gpu.to("minigpu")

        with torch.no_grad():
            expected = model_cpu(image)
            actual = model_gpu(self.to_minigpu(image))

        self.assert_tensor_close(actual, expected, rtol=2e-2, atol=2e-2)

    def test_tiny_mnist_conv2d_forward(self):
        model_cpu = torch.nn.Sequential(
            torch.nn.Conv2d(1, 2, kernel_size=3, stride=4, bias=True),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(2 * 7 * 7, 10, bias=True),
        ).eval()

        with torch.no_grad():
            model_cpu[0].weight.copy_(
                torch.linspace(-0.25, 0.25, 2 * 1 * 3 * 3, dtype=torch.float32).view(2, 1, 3, 3)
            )
            model_cpu[0].bias.copy_(torch.tensor([0.125, -0.25], dtype=torch.float32))
            model_cpu[3].weight.copy_(
                torch.linspace(-0.03, 0.03, 10 * 2 * 7 * 7, dtype=torch.float32).view(10, 2 * 7 * 7)
            )
            model_cpu[3].bias.copy_(torch.linspace(-0.1, 0.1, 10, dtype=torch.float32))

        image = torch.linspace(-1.0, 1.0, 28 * 28, dtype=torch.float32).view(1, 1, 28, 28)
        model_gpu = torch.nn.Sequential(
            torch.nn.Conv2d(1, 2, kernel_size=3, stride=4, bias=True),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(2 * 7 * 7, 10, bias=True),
        ).eval()
        model_gpu.load_state_dict(model_cpu.state_dict())
        model_gpu = model_gpu.to("minigpu")

        with torch.no_grad():
            expected = model_cpu(image)
            actual = model_gpu(self.to_minigpu(image))

        self.assert_tensor_close(actual, expected, rtol=3e-2, atol=3e-2)

    def test_linear_1d_kernel(self):
        x = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float32)
        weight = torch.tensor([[2.0, -1.0, 0.25], [-0.5, 3.0, 1.5]], dtype=torch.float32)

        self.assert_tensor_close(
            torch.nn.functional.linear(self.to_minigpu(x), self.to_minigpu(weight)),
            torch.nn.functional.linear(x, weight),
        )

    def test_conv1d_kernel(self):
        x = torch.tensor(
            [[[1.0, -2.0, 0.5, 3.0, -1.0], [0.25, 1.5, -0.5, 2.0, 1.0]]],
            dtype=torch.float32,
        )
        weight = torch.tensor(
            [[[0.5, -1.0, 0.25], [1.0, 0.5, -0.5]], [[-0.25, 0.75, 1.0], [0.5, -1.0, 0.25]]],
            dtype=torch.float32,
        )
        bias = torch.tensor([0.25, -0.5], dtype=torch.float32)

        self.assert_tensor_close(
            torch.nn.functional.conv1d(self.to_minigpu(x), self.to_minigpu(weight), stride=1),
            torch.nn.functional.conv1d(x, weight, stride=1),
        )
        self.assert_tensor_close(
            torch.nn.functional.conv1d(self.to_minigpu(x), self.to_minigpu(weight), self.to_minigpu(bias), stride=2),
            torch.nn.functional.conv1d(x, weight, bias, stride=2),
        )

    def test_conv2d_kernel(self):
        x = torch.tensor(
            [[[[1.0, -1.0, 2.0], [0.5, 3.0, -0.5], [2.0, -2.0, 1.5]]]],
            dtype=torch.float32,
        )
        weight = torch.tensor(
            [[[[0.5, -1.0], [1.5, 0.25]]], [[[-0.5, 0.75], [1.0, -0.25]]]],
            dtype=torch.float32,
        )
        bias = torch.tensor([0.125, -0.25], dtype=torch.float32)

        self.assert_tensor_close(
            torch.nn.functional.conv2d(self.to_minigpu(x), self.to_minigpu(weight), stride=1),
            torch.nn.functional.conv2d(x, weight, stride=1),
        )
        self.assert_tensor_close(
            torch.nn.functional.conv2d(self.to_minigpu(x), self.to_minigpu(weight), self.to_minigpu(bias), stride=1),
            torch.nn.functional.conv2d(x, weight, bias, stride=1),
        )


if __name__ == "__main__":
    unittest.main()
