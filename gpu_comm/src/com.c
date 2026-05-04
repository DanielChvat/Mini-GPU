/*
 * Packet Send format
 * [SOF 0xAA] [CMD 1B] [ADDR_HI 1B] [ADDR_LO 1B] [LEN 2B] [PAYLOAD LEN bytes] [CRC 1B]
 * CRC = XOR of every byte from CMD through last PAYLOAD byte (inclusive).
 */

 /*
 * ACK Format
 * [SOF 0xAA] [CMD 0x08|0x09] [X] [X] [00] [] [X]
 */

#include "com.h"
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

#define RETRIES 3

struct com_dev {
    fd_t fd;
    int timeout_ms;
};

static uint8_t com_crc_xor(const uint8_t *buf, size_t start, size_t end_exclusive) {
    uint8_t crc = 0;
    for (size_t i = start; i < end_exclusive; ++i) {
        crc ^= buf[i];
    }
    return crc;
}

static int com_build_packet_fields(uint8_t *buf,
                                    com_cmd_t cmd,
                                    uint16_t addr,
                                    uint16_t len_field,
                                    const uint8_t *payload,
                                    size_t payload_len) {
    if (!buf || (payload_len > 0 && !payload)) return COM_ERR_BAD_ARG;
    if (payload_len > COM_MAX_PAYLOAD) return COM_ERR_PAYLOAD_SIZE;

    buf[0] = COM_SOF;
    buf[1] = (uint8_t)cmd;
    buf[2] = (uint8_t)(addr >> 8);
    buf[3] = (uint8_t)(addr & 0xFFu);
    buf[4] = (uint8_t)(len_field >> 8);
    buf[5] = (uint8_t)(len_field & 0xFFu);

    if (payload_len > 0) {
        memcpy(buf + COM_HEADER_SIZE, payload, payload_len);
    }

    size_t crc_index = COM_HEADER_SIZE + payload_len;
    buf[crc_index] = com_crc_xor(buf, 1, crc_index);

    return (int)(crc_index + 1);
}

const char *com_strerror(com_err_t err){
    switch(err){
        case COM_OK                : return "OK";
        case COM_ERR_BAD_ARG       : return "Bad Argument (null pointer or out-of-range)";
        case COM_ERR_CRC           : return "CRC mismatch in received packet";
        case COM_ERR_NO_SOF        : return "Response missing start-of-frame byte (0xAA)";
        case COM_ERR_TIMEOUT       : return "Timed out waiting for response";
        case COM_ERR_SHORT         : return "Response shorter than expected";
        case COM_ERR_IO            : return "Serial I/O error";
        case COM_ERR_PAYLOAD_SIZE  : return "Payload exceeds 65535-byte max";
        case COM_ERR_OPEN          : return "Could not open serial port";
        case COM_ERR_FPGA          : return "FPGA encountered an error";
        default                     : return "Unknown Error Occurred"; 
    }
}

// Platform Specific Stuff
#ifdef _WIN32
com_dev_t *open_com(const char *port, int baud, int timeout_ms) {
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
 
    com_dev_t *dev = malloc(sizeof(*dev));
    if (!dev) { CloseHandle(h); return NULL; }
    dev->fd         = h;
    dev->timeout_ms = timeout_ms;
    return dev;
}
 
void close_com(com_dev_t *dev) {
    if (!dev) return;
    CloseHandle(dev->fd);
    free(dev);
}
 
com_err_t com_send_raw(com_dev_t *dev, const uint8_t *buf, size_t len) {
    if (!dev || !buf) return COM_ERR_BAD_ARG;
    DWORD written = 0;
    if (!WriteFile(dev->fd, buf, (DWORD)len, &written, NULL) || written != len)
        return COM_ERR_IO;
    return COM_OK;
}
 
com_err_t com_recv_raw(com_dev_t *dev, uint8_t *buf, size_t n) {
    if (!dev || !buf) return COM_ERR_BAD_ARG;
    size_t total = 0;
    while (total < n) {
        DWORD got = 0;
        if (!ReadFile(dev->fd, buf + total, (DWORD)(n - total), &got, NULL))
            return COM_ERR_IO;
        if (got == 0) return COM_ERR_TIMEOUT;
        total += got;
    }
    return COM_OK;
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
 
com_dev_t *open_com(const char *port, int baud, int timeout_ms) {
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
 
    com_dev_t *dev = malloc(sizeof(*dev));
    if (!dev) { close(fd); return NULL; }
    dev->fd         = fd;
    dev->timeout_ms = timeout_ms;
    return dev;
}
 
void close_com(com_dev_t *dev) {
    if (!dev) return;
    close(dev->fd);
    free(dev);
}
 
com_err_t com_send_raw(com_dev_t *dev, const uint8_t *buf, size_t len) {
    if (!dev || !buf) return COM_ERR_BAD_ARG;
    ssize_t n = write(dev->fd, buf, len);
    if (n < 0 || (size_t)n != len) return COM_ERR_IO;
    return COM_OK;
}
 
com_err_t com_recv_raw(com_dev_t *dev, uint8_t *buf, size_t n) {
    if (!dev || !buf) return COM_ERR_BAD_ARG;
    size_t total = 0;
    while (total < n) {
        ssize_t got = read(dev->fd, buf + total, n - total);
        if (got < 0)  return COM_ERR_IO;
        if (got == 0) return COM_ERR_TIMEOUT;
        total += (size_t)got;
    }
    return COM_OK;
}
 
#endif

int com_build_packet(uint8_t *buf,
                      com_cmd_t cmd,
                      uint16_t addr,
                      const uint8_t *payload,
                      uint16_t payload_len) {
    return com_build_packet_fields(buf, cmd, addr, payload_len, payload, payload_len);
}

static com_err_t com_send_ack(com_dev_t *dev, uint16_t addr){
    uint8_t tries = 0;
    com_err_t last_err = COM_ERR_IO;

    while (tries < RETRIES){
        uint8_t packet[COM_OVERHEAD];
        int packet_len = com_build_packet(packet, COM_CMD_ACK, addr, NULL, 0);
        if (packet_len < 0) return (com_err_t)packet_len;
        com_err_t err = com_send_raw(dev, packet, (size_t)packet_len);
        if (err == COM_OK) return COM_OK;
        last_err = err;
        tries++;
    }

    return last_err;
}

static com_err_t com_send_nak(com_dev_t *dev, uint16_t addr){
    uint8_t tries = 0;
    com_err_t last_err = COM_ERR_IO;

    while (tries < RETRIES){
        uint8_t packet[COM_OVERHEAD];
        int packet_len = com_build_packet(packet, COM_CMD_NAK, addr, NULL, 0);
        if (packet_len < 0) return (com_err_t)packet_len;
        com_err_t err = com_send_raw(dev, packet, (size_t)packet_len);
        if (err == COM_OK) return COM_OK;
        last_err = err;
        tries++;
    }

    return last_err;
}

com_err_t com_parse_packet(const uint8_t *buf, size_t buf_len, com_packet_t *pkt) {
    if (!buf || !pkt) return COM_ERR_BAD_ARG;
    if (buf_len < COM_OVERHEAD) return COM_ERR_SHORT;
    if (buf[0] != COM_SOF) return COM_ERR_NO_SOF;

    uint16_t payload_len = ((uint16_t)buf[4] << 8) | (uint16_t)buf[5];
    size_t expected_len = COM_OVERHEAD + (size_t)payload_len;
    if (buf_len < expected_len) return COM_ERR_SHORT;

    uint8_t expected_crc = com_crc_xor(buf, 1, expected_len - 1);
    if (buf[expected_len - 1] != expected_crc) return COM_ERR_CRC;

    pkt->cmd = buf[1];
    pkt->addr = ((uint16_t)buf[2] << 8) | (uint16_t)buf[3];
    pkt->len = payload_len;
    if (payload_len > 0) {
        memcpy(pkt->payload, buf + COM_HEADER_SIZE, payload_len);
    }

    return COM_OK;
}

com_err_t com_write_data(com_dev_t *dev, uint16_t addr,
                           const uint8_t *data, size_t data_len) {
    if (!dev || (data_len > 0 && !data)) return COM_ERR_BAD_ARG;
    if (data_len > (size_t)UINT16_MAX - (size_t)addr + 1u) return COM_ERR_BAD_ARG;

    uint8_t packet[COM_MAX_PACKET];
    size_t offset = 0;

    while (offset < data_len) {
        size_t remaining = data_len - offset;
        uint16_t chunk_len = (uint16_t)(remaining > COM_MAX_PAYLOAD
                             ? COM_MAX_PAYLOAD
                             : remaining);
        uint16_t chunk_addr = (uint16_t)(addr + offset);

        int packet_len = com_build_packet(packet,
                                           COM_CMD_WRITE_DATA,
                                           chunk_addr,
                                           data + offset,
                                           chunk_len);
        if (packet_len < 0) return (com_err_t)packet_len;

        com_err_t last_err = COM_ERR_FPGA;
        int success = 0;

        for (int attempt = 0; attempt < RETRIES; attempt++) {
            com_err_t err = com_send_raw(dev, packet, (size_t)packet_len);
            if (err != COM_OK) {
                last_err = err;
                continue;
            }

            uint8_t ack_buf[COM_OVERHEAD];
            com_packet_t ack_packet;

            err = com_recv_raw(dev, ack_buf, COM_OVERHEAD);
            if (err != COM_OK) {
                last_err = err;
                continue;
            }

            err = com_parse_packet(ack_buf, COM_OVERHEAD, &ack_packet);
            if (err != COM_OK) {
                last_err = err;
                continue;
            }

            if (ack_packet.cmd != COM_CMD_ACK || ack_packet.addr != chunk_addr || ack_packet.len != 0) {
                last_err = COM_ERR_FPGA;
                continue;
            }

            success = 1;
            break;
        }

        if (!success) return last_err;

        offset += chunk_len;
    }

    return COM_OK;
}

com_err_t com_read_data(com_dev_t *dev, uint16_t addr,
                          uint8_t *out_buf, size_t data_len) {
    if (!dev || (data_len > 0 && !out_buf)) return COM_ERR_BAD_ARG;
    if (data_len > (size_t)UINT16_MAX - (size_t)addr + 1u) return COM_ERR_BAD_ARG;

    uint8_t tx_packet[COM_OVERHEAD];
    uint8_t rx_packet[COM_MAX_PACKET];
    size_t offset = 0;

    while (offset < data_len) {
        size_t remaining = data_len - offset;
        uint16_t chunk_len = (uint16_t)(remaining > COM_MAX_PAYLOAD
                             ? COM_MAX_PAYLOAD
                             : remaining);
        uint16_t chunk_addr = (uint16_t)(addr + offset);

        int packet_len = com_build_packet_fields(tx_packet,
                                                  COM_CMD_READ_DATA,
                                                  chunk_addr,
                                                  chunk_len,
                                                  NULL,
                                                  0);
        if (packet_len < 0) return (com_err_t)packet_len;

        com_err_t last_err = COM_ERR_FPGA;
        int success = 0;

        for (int attempt = 0; attempt < RETRIES; attempt++) {
            com_err_t err = com_send_raw(dev, tx_packet, (size_t)packet_len);
            if (err != COM_OK) {
                last_err = err;
                continue;
            }

            err = com_recv_raw(dev, rx_packet, COM_HEADER_SIZE);
            if (err != COM_OK) {
                last_err = err;
                continue;
            }

            if (rx_packet[0] != COM_SOF) {
                last_err = COM_ERR_NO_SOF;
                (void)com_send_nak(dev, chunk_addr);
                continue;
            }

            uint16_t response_len = ((uint16_t)rx_packet[4] << 8) | (uint16_t)rx_packet[5];
            if (response_len > COM_MAX_PAYLOAD) {
                last_err = COM_ERR_PAYLOAD_SIZE;
                (void)com_send_nak(dev, chunk_addr);
                continue;
            }

            err = com_recv_raw(dev,
                                rx_packet + COM_HEADER_SIZE,
                                (size_t)response_len + 1u);
            if (err != COM_OK) {
                last_err = err;
                (void)com_send_nak(dev, chunk_addr);
                continue;
            }

            com_packet_t response;
            err = com_parse_packet(rx_packet,
                                    COM_OVERHEAD + (size_t)response_len,
                                    &response);
            if (err != COM_OK) {
                last_err = err;
                (void)com_send_nak(dev, chunk_addr);
                continue;
            }

            if (response.len != chunk_len) {
                last_err = COM_ERR_SHORT;
                (void)com_send_nak(dev, chunk_addr);
                continue;
            }

            if (response.cmd != COM_CMD_READ_DATA || response.addr != chunk_addr) {
                last_err = COM_ERR_FPGA;
                (void)com_send_nak(dev, chunk_addr);
                continue;
            }

            err = com_send_ack(dev, chunk_addr);
            if (err != COM_OK) {
                last_err = err;
                continue;
            }

            memcpy(out_buf + offset, response.payload, response_len);
            success = 1;
            break;
        }

        if (!success) return last_err;

        offset += chunk_len;
    }

    return COM_OK;
}
