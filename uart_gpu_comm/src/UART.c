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

const char *uart_strerror(uart_err_t err){
    switch(err){
        case UART_OK                : return "OK";
        case UART_ERR_BAD_ARG       : return "Bad Argument (null pointer or out-of-range)";
        case UART_ERR_CRC           : return "CRC mismatch in recieved packet";
        case UART_ERR_NO_SOF        : return "Response missing start-of-frame byte (0xAA)";
        case UART_ERR_TIMEOUT       : return "Timed out waiting for response";
        case UART_ERR_SHORT         : return "Response shorer than expected";
        case UART_ERR_IO            : return "Serial I/O error";
        case UART_ERR_PAYLOAD_SIZE  : return "Payload exceeds 65536-byte max";
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
        return SIMT_ERR_IO;
    return SIMT_OK;
}
 
uart_err_t uart_recv_raw(uart_dev_t *dev, uint8_t *buf, size_t n) {
    if (!dev || !buf) return UART_ERR_BAD_ARG;
    size_t total = 0;
    while (total < n) {
        DWORD got = 0;
        if (!ReadFile(dev->fd, buf + total, (DWORD)(n - total), &got, NULL))
            return SIMT_ERR_IO;
        if (got == 0) return UART_ERR_TIMEOUT;
        total += got;
    }
    return UART_OK;
}
 
#else  /* POSIX */
 
static speed_t baud_to_speed(int baud) {
    switch (baud) {
        case 9600:   return B9600;
        case 19200:  return B19200;
        case 38400:  return B38400;
        case 57600:  return B57600;
        case 115200: return B115200;
        case 230400: return B230400;
        case 460800: return B460800;
        case 921600: return B921600;
        default:     return B115200;
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
