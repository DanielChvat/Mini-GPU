#ifndef UART_H
#define UART_H

#ifdef __cplusplus
extern "C" {
#endif

#include "CONSTANTS.h"
#include <stdint.h>
#include <stddef.h>
typedef enum {
    UART_CMD_WRITE_DATA    = 0x01,  /* write LEN bytes to data BRAM at ADDR  */
    UART_CMD_READ_DATA     = 0x02,  /* request LEN bytes from ADDR           */
    UART_CMD_WRITE_PROGRAM = 0x03,  /* write LEN bytes to instr BRAM at ADDR */
    UART_CMD_LAUNCH        = 0x04,  /* start execution (ADDR = base PC)      */
    UART_CMD_READ_STATUS   = 0x05,  /* read status register (no payload)     */
    UART_CMD_WRITE_HASH    = 0x06,  /* load expected SHA256 hash (32 bytes)  */
    UART_CMD_VALIDATE      = 0x07,  /* trigger hash validation               */
} uart_cmd_t;

typedef enum {
    UART_OK                =  0,
    UART_ERR_BAD_ARG       = -1,  /* null pointer or out-of-range argument  */
    UART_ERR_CRC           = -2,  /* received packet has wrong CRC          */
    UART_ERR_NO_SOF        = -3,  /* response did not start with 0xAA       */
    UART_ERR_TIMEOUT       = -4,  /* read() returned 0 bytes                */
    UART_ERR_SHORT         = -5,  /* response shorter than expected         */
    UART_ERR_IO            = -6,  /* OS-level read/write failure            */
    UART_ERR_PAYLOAD_SIZE  = -7,  /* payload exceeds UART_MAX_PAYLOAD       */
    UART_ERR_OPEN          = -8,  /* could not open serial port             */
    UART_ERR_FPGA          = -9,  /* FPGA returned an error status          */
} uart_err_t;

typedef struct {
    uint8_t cmd;
    uint16_t addr;
    uint16_t len;
    uint8_t payload[UART_MAX_PAYLOAD];
} uart_packet_t;

typedef struct uart_dev uart_dev_t;

/* Open a serial port and return a heap-allocated device handle.
 * port    : e.g. "/dev/ttyUSB0" on Linux, "COM3" on Windows
 * baud    : baud rate, e.g. 115200
 * timeout_ms: per-byte read timeout in milliseconds
 * Returns NULL on failure. */
uart_dev_t *open_uart(const char *port, int baud, int timeout_ms);

/*Close the serial port and free the handle*/
void close_uart(uart_dev_t *dev);

/*Convert Error Code into human-readable string*/
const char *uart_strerror(uart_err_t err);

/* Send a pre-built packet. Returns UART_OK or uart_err_t. */
uart_err_t uart_send_raw(uart_dev_t *dev, const uint8_t *buf, size_t len);

/* Read exactly n bytes into buf. Returns UART_OK or uart_err_t. */
uart_err_t uart_recv_raw(uart_dev_t *dev, uint8_t *buf, size_t n);

/* Build a packet into buf and returns total packet length on success or uart_err_t*/
int uart_build_packet(uint8_t *buf, uart_cmd_t cmd, uint16_t addr, const uint8_t *payload, uint16_t payload_len);

/* Parse a packet from buf into pkt
 * Verifies SOF, CRC, returns UART_OK uart_err_t
*/
uart_err_t uart_parse_packet(const uint8_t *buf, size_t buf_len, uart_packet_t *pkt);

/* Write up to UART_MAX_PAYLOAD bytes of data to data BRAM at addr.
 * Splits automatically into multiple packets if data_len > UART_MAX_PAYLOAD. */
uart_err_t uart_write_data(uart_dev_t *dev, uint16_t addr,
                           const uint8_t *data, size_t data_len);

/* Read data_len bytes from data BRAM starting at addr into out_buf. */
uart_err_t uart_read_data(uart_dev_t *dev, uint16_t addr,
                          uint8_t *out_buf, size_t data_len);

#ifdef __cplusplus
}
#endif

#endif
