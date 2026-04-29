#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "com.h"

#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

namespace py = pybind11;

static void throw_if_com_error(com_err_t err) {
    if (err != COM_OK) {
        throw std::runtime_error(com_strerror(err));
    }
}

static std::string bytes_to_string(py::bytes data) {
    char *buffer = nullptr;
    ssize_t length = 0;

    if (PYBIND11_BYTES_AS_STRING_AND_SIZE(data.ptr(), &buffer, &length)) {
        throw std::runtime_error("failed to parse bytes object");
    }

    if (length < 0) {
        throw std::runtime_error("invalid bytes length");
    }

    return std::string(buffer, static_cast<size_t>(length));
}

class UartDevice {
public:
    UartDevice(const std::string& device_name, int baud, int timeout_ms) {
        dev_ = open_com(device_name.c_str(), baud, timeout_ms);

        if (!dev_) {
            throw std::runtime_error("failed to open COM device");
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
            close_com(dev_);
            dev_ = nullptr;
        }
    }

    bool is_open() const {
        return dev_ != nullptr;
    }

    void send_raw(py::bytes data) {
        if (!dev_) {
            throw std::runtime_error("COM device is closed");
        }

        std::string buffer = bytes_to_string(data);

        com_err_t err = com_send_raw(
            dev_,
            reinterpret_cast<const uint8_t *>(buffer.data()),
            buffer.size()
        );

        throw_if_com_error(err);
    }

    py::bytes recv_raw(size_t n) {
        if (!dev_) {
            throw std::runtime_error("COM device is closed");
        }

        std::string buffer;
        buffer.resize(n);

        com_err_t err = com_recv_raw(
            dev_,
            reinterpret_cast<uint8_t *>(buffer.data()),
            n
        );

        throw_if_com_error(err);

        return py::bytes(buffer);
    }

    void write_data(uint16_t addr, py::bytes data) {
        if (!dev_) {
            throw std::runtime_error("COM device is closed");
        }

        std::string buffer = bytes_to_string(data);

        com_err_t err = com_write_data(
            dev_,
            addr,
            reinterpret_cast<const uint8_t *>(buffer.data()),
            buffer.size()
        );

        throw_if_com_error(err);
    }

    py::bytes read_data(uint16_t addr, size_t n) {
        if (!dev_) {
            throw std::runtime_error("COM device is closed");
        }

        std::string buffer;
        buffer.resize(n);

        com_err_t err = com_read_data(
            dev_,
            addr,
            reinterpret_cast<uint8_t *>(buffer.data()),
            n
        );

        throw_if_com_error(err);

        return py::bytes(buffer);
    }

private:
    com_dev_t *dev_ = nullptr;
};

PYBIND11_MODULE(_gpu_comm, m) {
    m.doc() = "Python bindings for COM GPU communication";
    m.attr("MAX_PAYLOAD") = COM_MAX_PAYLOAD;

    py::enum_<com_err_t>(m, "Error", py::arithmetic())
        .value("OK", COM_OK)
        .value("BAD_ARG", COM_ERR_BAD_ARG)
        .value("CRC", COM_ERR_CRC)
        .value("NO_SOF", COM_ERR_NO_SOF)
        .value("TIMEOUT", COM_ERR_TIMEOUT)
        .value("SHORT", COM_ERR_SHORT)
        .value("IO", COM_ERR_IO)
        .value("PAYLOAD_SIZE", COM_ERR_PAYLOAD_SIZE)
        .value("OPEN", COM_ERR_OPEN)
        .value("FPGA", COM_ERR_FPGA);

    py::enum_<com_cmd_t>(m, "Command", py::arithmetic())
        .value("WRITE_DATA", COM_CMD_WRITE_DATA)
        .value("READ_DATA", COM_CMD_READ_DATA)
        .value("WRITE_PROGRAM", COM_CMD_WRITE_PROGRAM)
        .value("LAUNCH", COM_CMD_LAUNCH)
        .value("READ_STATUS", COM_CMD_READ_STATUS)
        .value("WRITE_HASH", COM_CMD_WRITE_HASH)
        .value("VALIDATE", COM_CMD_VALIDATE);

    m.def("strerror",
          [](int err) {
              return com_strerror(static_cast<com_err_t>(err));
          },
          "Convert COM error code to string");

    m.def("build_packet",
          [](com_cmd_t cmd, uint16_t addr, py::bytes payload) {
              std::string payload_buffer = bytes_to_string(payload);
              if (payload_buffer.size() > UINT16_MAX) {
                  throw std::runtime_error(com_strerror(COM_ERR_PAYLOAD_SIZE));
              }

              std::string packet;
              packet.resize(COM_OVERHEAD + payload_buffer.size());

              int packet_len = com_build_packet(
                  reinterpret_cast<uint8_t *>(packet.data()),
                  cmd,
                  addr,
                  reinterpret_cast<const uint8_t *>(payload_buffer.data()),
                  static_cast<uint16_t>(payload_buffer.size())
              );

              if (packet_len < 0) {
                  throw_if_com_error(static_cast<com_err_t>(packet_len));
              }

              packet.resize(static_cast<size_t>(packet_len));
              return py::bytes(packet);
          },
          py::arg("cmd"),
          py::arg("addr"),
          py::arg("payload") = py::bytes(),
          "Build a COM packet");

    m.def("parse_packet",
          [](py::bytes data) {
              std::string buffer = bytes_to_string(data);
              com_packet_t packet;

              com_err_t err = com_parse_packet(
                  reinterpret_cast<const uint8_t *>(buffer.data()),
                  buffer.size(),
                  &packet
              );
              throw_if_com_error(err);

              py::dict result;
              result["cmd"] = packet.cmd;
              result["addr"] = packet.addr;
              result["len"] = packet.len;
              result["payload"] = py::bytes(
                  reinterpret_cast<const char *>(packet.payload),
                  packet.len
              );
              return result;
          },
          py::arg("data"),
          "Parse and validate a COM packet");

    py::class_<UartDevice>(m, "Device")
        .def(py::init<const std::string&, int, int>(),
             py::arg("device_name"),
             py::arg("baud"),
             py::arg("timeout_ms") = 1000)
        .def("close", &UartDevice::close)
        .def("is_open", &UartDevice::is_open)
        .def("send_raw", &UartDevice::send_raw,
             py::arg("data"),
             "Send raw bytes over COM")
        .def("recv_raw", &UartDevice::recv_raw, py::arg("n"))
        .def("write_data", &UartDevice::write_data,
             py::arg("addr"),
             py::arg("data"),
             "Write bytes to data BRAM")
        .def("read_data", &UartDevice::read_data,
             py::arg("addr"),
             py::arg("n"),
             "Read bytes from data BRAM");
}
