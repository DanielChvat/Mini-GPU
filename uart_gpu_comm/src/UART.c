/*
 * Packet format
 * [SOF 0xAA] [CMD 1B] [ADDR_HI 1B] [ADDR_LO 1B] [LEN 2B] [PAYLOAD LEN bytes] [CRC 1B]
 * CRC = XOR of every byte from CMD through last PAYLOAD byte (inclusive).
 */

#include "UART.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#ifdef _WIN32
#include <windows.h>
typedef HANDLE fd_t;
#define FD_INVALID INVALID_HANDLE_VALUE
#else
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <time.h>
typedef int fd_t;
#define FD_INVALID (-1)
#endif

struct uart_dev {
    fd_t fd;
    int timeout_ms;
};

static uint8_t uart_crc_xor(const uint8_t *buf, size_t start, size_t end_exclusive) {
    uint8_t crc = 0;
    for (size_t i = start; i < end_exclusive; ++i) {
        crc ^= buf[i];
    }
    return crc;
}

static int uart_build_packet_fields(uint8_t *buf,
                                    uart_cmd_t cmd,
                                    uint16_t addr,
                                    uint16_t len_field,
                                    const uint8_t *payload,
                                    size_t payload_len) {
    if (!buf || (payload_len > 0 && !payload)) return UART_ERR_BAD_ARG;
    if (payload_len > UART_MAX_PAYLOAD) return UART_ERR_PAYLOAD_SIZE;

    buf[0] = UART_SOF;
    buf[1] = (uint8_t)cmd;
    buf[2] = (uint8_t)(addr >> 8);
    buf[3] = (uint8_t)(addr & 0xFFu);
    buf[4] = (uint8_t)(len_field >> 8);
    buf[5] = (uint8_t)(len_field & 0xFFu);

    if (payload_len > 0) {
        memcpy(buf + UART_HEADER_SIZE, payload, payload_len);
    }

    size_t crc_index = UART_HEADER_SIZE + payload_len;
    buf[crc_index] = uart_crc_xor(buf, 1, crc_index);

    return (int)(crc_index + 1);
}

const char *uart_strerror(uart_err_t err){
    switch(err){
        case UART_OK                : return "OK";
        case UART_ERR_BAD_ARG       : return "Bad Argument (null pointer or out-of-range)";
        case UART_ERR_CRC           : return "CRC mismatch in received packet";
        case UART_ERR_NO_SOF        : return "Response missing start-of-frame byte (0xAA)";
        case UART_ERR_TIMEOUT       : return "Timed out waiting for response";
        case UART_ERR_SHORT         : return "Response shorter than expected";
        case UART_ERR_IO            : return "Serial I/O error";
        case UART_ERR_PAYLOAD_SIZE  : return "Payload exceeds 65535-byte max";
        case UART_ERR_OPEN          : return "Could not open serial port";
        case UART_ERR_FPGA          : return "FPGA encountered an error";
        default                     : return "Unknown Error Occurred"; 
    }
}

// Platform Specific Stuff
#ifdef _WIN32
uart_dev_t *open_uart(const char *port, int baud, int timeout_ms) {
    char full_port[64];
    snprintf(full_port, sizeof(full_port), "\\\\.\\%s", port);
 
    HANDLE h = CreateFileA(full_port, GENERIC_READ | GENERIC_WRITE,
                           0, NULL, OPEN_EXISTING, 0, NULL);
    if (h == INVALID_HANDLE_VALUE) return NULL;
 
    DCB dcb = {0};
    dcb.DCBlength = sizeof(dcb);
    GetCommState(h, &dcb);
    dcb.BaudRate = (DWORD)baud;
    dcb.ByteSize = 8;
    dcb.Parity   = NOPARITY;
    dcb.StopBits = ONESTOPBIT;
    SetCommState(h, &dcb);
 
    COMMTIMEOUTS to = {0};
    to.ReadIntervalTimeout         = (DWORD)timeout_ms;
    to.ReadTotalTimeoutMultiplier  = (DWORD)timeout_ms;
    to.ReadTotalTimeoutConstant    = (DWORD)timeout_ms;
    SetCommTimeouts(h, &to);
 
    uart_dev_t *dev = malloc(sizeof(*dev));
    if (!dev) { CloseHandle(h); return NULL; }
    dev->fd         = h;
    dev->timeout_ms = timeout_ms;
    return dev;
}
 
void close_uart(uart_dev_t *dev) {
    if (!dev) return;
    CloseHandle(dev->fd);
    free(dev);
}
 
uart_err_t uart_send_raw(uart_dev_t *dev, const uint8_t *buf, size_t len) {
    if (!dev || !buf) return UART_ERR_BAD_ARG;
    DWORD written = 0;
    if (!WriteFile(dev->fd, buf, (DWORD)len, &written, NULL) || written != len)
        return UART_ERR_IO;
    return UART_OK;
}
 
uart_err_t uart_recv_raw(uart_dev_t *dev, uint8_t *buf, size_t n) {
    if (!dev || !buf) return UART_ERR_BAD_ARG;
    size_t total = 0;
    while (total < n) {
        DWORD got = 0;
        if (!ReadFile(dev->fd, buf + total, (DWORD)(n - total), &got, NULL))
            return UART_ERR_IO;
        if (got == 0) return UART_ERR_TIMEOUT;
        total += got;
    }
    return UART_OK;
}
 
#else  /* POSIX */
 
static speed_t baud_to_speed(int baud) {
switch (baud) {
        case 9600: return B9600;
        case 19200: return B19200;
        case 38400: return B38400;
        case 57600: return B57600;
        case 115200: return B115200;
        case 230400: return B230400;

#ifdef B460800
        case 460800: return B460800;
#endif

#ifdef B921600
        case 921600: return B921600;
#endif

        default:
            errno = EINVAL;
            fprintf(stderr, "Unsupported baud rate on this platform: %d\n", baud);
            return -1;
    }
}
 
uart_dev_t *open_uart(const char *port, int baud, int timeout_ms) {
    int fd = open(port, O_RDWR | O_NOCTTY | O_SYNC);
    if (fd < 0) return NULL;
 
    struct termios tty;
    memset(&tty, 0, sizeof(tty));
    tcgetattr(fd, &tty);
 
    speed_t sp = baud_to_speed(baud);
    cfsetospeed(&tty, sp);
    cfsetispeed(&tty, sp);
 
    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;  /* 8-bit chars          */
    tty.c_cflag |= (CLOCAL | CREAD);              /* ignore modem ctrl    */
    tty.c_cflag &= ~(PARENB | PARODD);            /* no parity            */
    tty.c_cflag &= ~CSTOPB;                        /* 1 stop bit           */
    tty.c_cflag &= ~CRTSCTS;                       /* no flow control      */
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);       /* no software flow ctrl*/
    tty.c_iflag &= ~(IGNBRK | BRKINT | ICRNL | INLCR | PARMRK | INPCK | ISTRIP);
    tty.c_oflag  = 0;
    tty.c_lflag  = 0;
 
    /* VMIN=0, VTIME = timeout in 0.1s increments (minimum 1 = 100ms) */
    int vtime = (timeout_ms + 99) / 100;
    if (vtime < 1) vtime = 1;
    if (vtime > 255) vtime = 255;
    tty.c_cc[VMIN]  = 0;
    tty.c_cc[VTIME] = (cc_t)vtime;
 
    tcsetattr(fd, TCSANOW, &tty);
    tcflush(fd, TCIOFLUSH);
 
    uart_dev_t *dev = malloc(sizeof(*dev));
    if (!dev) { close(fd); return NULL; }
    dev->fd         = fd;
    dev->timeout_ms = timeout_ms;
    return dev;
}
 
void close_uart(uart_dev_t *dev) {
    if (!dev) return;
    close(dev->fd);
    free(dev);
}
 
uart_err_t uart_send_raw(uart_dev_t *dev, const uint8_t *buf, size_t len) {
    if (!dev || !buf) return UART_ERR_BAD_ARG;
    ssize_t n = write(dev->fd, buf, len);
    if (n < 0 || (size_t)n != len) return UART_ERR_IO;
    return UART_OK;
}
 
uart_err_t uart_recv_raw(uart_dev_t *dev, uint8_t *buf, size_t n) {
    if (!dev || !buf) return UART_ERR_BAD_ARG;
    size_t total = 0;
    while (total < n) {
        ssize_t got = read(dev->fd, buf + total, n - total);
        if (got < 0)  return UART_ERR_IO;
        if (got == 0) return UART_ERR_TIMEOUT;
        total += (size_t)got;
    }
    return UART_OK;
}
 
#endif

int uart_build_packet(uint8_t *buf,
                      uart_cmd_t cmd,
                      uint16_t addr,
                      const uint8_t *payload,
                      uint16_t payload_len) {
    return uart_build_packet_fields(buf, cmd, addr, payload_len, payload, payload_len);
}

uart_err_t uart_parse_packet(const uint8_t *buf, size_t buf_len, uart_packet_t *pkt) {
    if (!buf || !pkt) return UART_ERR_BAD_ARG;
    if (buf_len < UART_OVERHEAD) return UART_ERR_SHORT;
    if (buf[0] != UART_SOF) return UART_ERR_NO_SOF;

    uint16_t payload_len = ((uint16_t)buf[4] << 8) | (uint16_t)buf[5];
    size_t expected_len = UART_OVERHEAD + (size_t)payload_len;
    if (buf_len < expected_len) return UART_ERR_SHORT;

    uint8_t expected_crc = uart_crc_xor(buf, 1, expected_len - 1);
    if (buf[expected_len - 1] != expected_crc) return UART_ERR_CRC;

    pkt->cmd = buf[1];
    pkt->addr = ((uint16_t)buf[2] << 8) | (uint16_t)buf[3];
    pkt->len = payload_len;
    if (payload_len > 0) {
        memcpy(pkt->payload, buf + UART_HEADER_SIZE, payload_len);
    }

    return UART_OK;
}

uart_err_t uart_write_data(uart_dev_t *dev, uint16_t addr,
                           const uint8_t *data, size_t data_len) {
    if (!dev || (data_len > 0 && !data)) return UART_ERR_BAD_ARG;
    if (data_len > (size_t)UINT16_MAX - (size_t)addr + 1u) return UART_ERR_BAD_ARG;

    uint8_t packet[UART_MAX_PACKET];
    size_t offset = 0;

    while (offset < data_len) {
        size_t remaining = data_len - offset;
        uint16_t chunk_len = (uint16_t)(remaining > UART_MAX_PAYLOAD
                             ? UART_MAX_PAYLOAD
                             : remaining);
        uint16_t chunk_addr = (uint16_t)(addr + offset);

        int packet_len = uart_build_packet(packet,
                                           UART_CMD_WRITE_DATA,
                                           chunk_addr,
                                           data + offset,
                                           chunk_len);
        if (packet_len < 0) return (uart_err_t)packet_len;

        uart_err_t err = uart_send_raw(dev, packet, (size_t)packet_len);
        if (err != UART_OK) return err;

        offset += chunk_len;
    }

    return UART_OK;
}

uart_err_t uart_read_data(uart_dev_t *dev, uint16_t addr,
                          uint8_t *out_buf, size_t data_len) {
    if (!dev || (data_len > 0 && !out_buf)) return UART_ERR_BAD_ARG;
    if (data_len > (size_t)UINT16_MAX - (size_t)addr + 1u) return UART_ERR_BAD_ARG;

    uint8_t tx_packet[UART_OVERHEAD];
    uint8_t rx_packet[UART_MAX_PACKET];
    size_t offset = 0;

    while (offset < data_len) {
        size_t remaining = data_len - offset;
        uint16_t chunk_len = (uint16_t)(remaining > UART_MAX_PAYLOAD
                             ? UART_MAX_PAYLOAD
                             : remaining);
        uint16_t chunk_addr = (uint16_t)(addr + offset);

        int packet_len = uart_build_packet_fields(tx_packet,
                                                  UART_CMD_READ_DATA,
                                                  chunk_addr,
                                                  chunk_len,
                                                  NULL,
                                                  0);
        if (packet_len < 0) return (uart_err_t)packet_len;

        uart_err_t err = uart_send_raw(dev, tx_packet, (size_t)packet_len);
        if (err != UART_OK) return err;

        err = uart_recv_raw(dev, rx_packet, UART_HEADER_SIZE);
        if (err != UART_OK) return err;
        if (rx_packet[0] != UART_SOF) return UART_ERR_NO_SOF;

        uint16_t response_len = ((uint16_t)rx_packet[4] << 8) | (uint16_t)rx_packet[5];
        if (response_len != chunk_len) return UART_ERR_SHORT;

        err = uart_recv_raw(dev,
                            rx_packet + UART_HEADER_SIZE,
                            (size_t)response_len + 1u);
        if (err != UART_OK) return err;

        uart_packet_t response;
        err = uart_parse_packet(rx_packet,
                                UART_OVERHEAD + (size_t)response_len,
                                &response);
        if (err != UART_OK) return err;
        if (response.cmd != UART_CMD_READ_DATA || response.addr != chunk_addr) {
            return UART_ERR_FPGA;
        }

        memcpy(out_buf + offset, response.payload, response_len);
        offset += response_len;
    }

    return UART_OK;
}
