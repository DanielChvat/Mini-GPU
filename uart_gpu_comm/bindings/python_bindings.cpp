#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <stdexcept>
#include <string>

#include "UART.h"

namespace py = pybind11;

static void throw_if_uart_error(uart_err_t err) {
    if (err != UART_OK) {
        throw std::runtime_error(uart_strerror(err));
    }
}

class UartDevice {
public:
    UartDevice(const std::string& device_name, int baud, int timeout_ms) {
        dev_ = open_uart(device_name.c_str(), baud, timeout_ms);

        if (!dev_) {
            throw std::runtime_error("failed to open UART device");
        }
    }

    ~UartDevice() {
        close();
    }

    UartDevice(const UartDevice&) = delete;
    UartDevice& operator=(const UartDevice&) = delete;

    UartDevice(UartDevice&& other) noexcept {
        dev_ = other.dev_;
        other.dev_ = nullptr;
    }

    UartDevice& operator=(UartDevice&& other) noexcept {
        if (this != &other) {
            close();
            dev_ = other.dev_;
            other.dev_ = nullptr;
        }

        return *this;
    }

    void close() {
        if (dev_) {
            close_uart(dev_);
            dev_ = nullptr;
        }
    }

    bool is_open() const {
        return dev_ != nullptr;
    }

    void send_raw(py::bytes data) {
        if (!dev_) {
            throw std::runtime_error("UART device is closed");
        }

        char *buffer = nullptr;
        ssize_t length = 0;

        if (PYBIND11_BYTES_AS_STRING_AND_SIZE(data.ptr(), &buffer, &length)) {
            throw std::runtime_error("failed to parse bytes object");
        }

        if (length < 0) {
            throw std::runtime_error("invalid bytes length");
        }

        uart_err_t err = uart_send_raw(
            dev_,
            reinterpret_cast<const uint8_t *>(buffer),
            static_cast<size_t>(length)
        );

        throw_if_uart_error(err);
    }

    py::bytes recv_raw(size_t n) {
        if (!dev_) {
            throw std::runtime_error("UART device is closed");
        }

        std::string buffer;
        buffer.resize(n);

        uart_err_t err = uart_recv_raw(
            dev_,
            reinterpret_cast<uint8_t *>(buffer.data()),
            n
        );

        throw_if_uart_error(err);

        return py::bytes(buffer);
    }

private:
    uart_dev_t *dev_ = nullptr;
};

PYBIND11_MODULE(_uart_gpu_comm, m) {
    m.doc() = "Python bindings for UART GPU communication";

    py::enum_<uart_err_t>(m, "Error", py::arithmetic())
        .value("OK", UART_OK)
        .value("BAD_ARG", UART_ERR_BAD_ARG)
        .value("CRC", UART_ERR_CRC)
        .value("NO_SOF", UART_ERR_NO_SOF)
        .value("TIMEOUT", UART_ERR_TIMEOUT)
        .value("SHORT", UART_ERR_SHORT)
        .value("IO", UART_ERR_IO)
        .value("PAYLOAD_SIZE", UART_ERR_PAYLOAD_SIZE)
        .value("OPEN", UART_ERR_OPEN)
        .value("FPGA", UART_ERR_FPGA);

    m.def("strerror",
          [](int err) {
              return uart_strerror(static_cast<uart_err_t>(err));
          },
          "Convert UART error code to string");

    py::class_<UartDevice>(m, "Device")
        .def(py::init<const std::string&, int, int>(),
             py::arg("device_name"),
             py::arg("baud"),
             py::arg("timeout_ms") = 1000)
        .def("close", &UartDevice::close)
        .def("is_open", &UartDevice::is_open)
        .def("send_raw", &UartDevice::send_raw,
             py::arg("data"),
             "Send raw bytes over UART")
        .def("recv_raw", &UartDevice::recv_raw, py::arg("n"));
}