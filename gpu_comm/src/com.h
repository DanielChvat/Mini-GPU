#ifndef COM_H
#define COM_H

#ifdef __cplusplus
extern "C" {
#endif

#include "constants.h"
#include <stdint.h>
#include <stddef.h>
typedef enum {
    COM_CMD_WRITE_DATA    = 0x01,  /* write LEN bytes to data BRAM at ADDR  */
    COM_CMD_READ_DATA     = 0x02,  /* request LEN bytes from ADDR           */
    COM_CMD_WRITE_PROGRAM = 0x03,  /* write LEN bytes to instr BRAM at ADDR */
    COM_CMD_LAUNCH        = 0x04,  /* start execution (ADDR = base PC)      */
    COM_CMD_READ_STATUS   = 0x05,  /* read status register (no payload)     */
    COM_CMD_WRITE_HASH    = 0x06,  /* load expected SHA256 hash (32 bytes)  */
    COM_CMD_VALIDATE      = 0x07,  /* trigger hash validation               */
    COM_CMD_ACK           = 0x08,  /* data was recieved properly            */
    COM_CMD_NAK           = 0x09,  /* data was not recieved properly.       */
} com_cmd_t;

typedef enum {
    COM_OK                =  0,
    COM_ERR_BAD_ARG       = -1,  /* null pointer or out-of-range argument  */
    COM_ERR_CRC           = -2,  /* received packet has wrong CRC          */
    COM_ERR_NO_SOF        = -3,  /* response did not start with 0xAA       */
    COM_ERR_TIMEOUT       = -4,  /* read() returned 0 bytes                */
    COM_ERR_SHORT         = -5,  /* response shorter than expected         */
    COM_ERR_IO            = -6,  /* OS-level read/write failure            */
    COM_ERR_PAYLOAD_SIZE  = -7,  /* payload exceeds COM_MAX_PAYLOAD       */
    COM_ERR_OPEN          = -8,  /* could not open serial port             */
    COM_ERR_FPGA          = -9,  /* FPGA returned an error status          */
} com_err_t;

typedef struct {
    uint8_t cmd;
    uint16_t addr;
    uint16_t len;
    uint8_t payload[COM_MAX_PAYLOAD];
} com_packet_t;

typedef struct com_dev com_dev_t;

/* Open a serial port and return a heap-allocated device handle.
 * port    : e.g. "/dev/ttyUSB0" on Linux, "COM3" on Windows
 * baud    : baud rate, e.g. 115200
 * timeout_ms: per-byte read timeout in milliseconds
 * Returns NULL on failure. */
com_dev_t *open_com(const char *port, int baud, int timeout_ms);

/*Close the serial port and free the handle*/
void close_com(com_dev_t *dev);

/*Convert Error Code into human-readable string*/
const char *com_strerror(com_err_t err);

/* Send a pre-built packet. Returns COM_OK or com_err_t. */
com_err_t com_send_raw(com_dev_t *dev, const uint8_t *buf, size_t len);

/* Read exactly n bytes into buf. Returns COM_OK or com_err_t. */
com_err_t com_recv_raw(com_dev_t *dev, uint8_t *buf, size_t n);

/* Build a packet into buf and returns total packet length on success or com_err_t*/
int com_build_packet(uint8_t *buf, com_cmd_t cmd, uint16_t addr, const uint8_t *payload, uint16_t payload_len);

/* Parse a packet from buf into pkt
 * Verifies SOF, CRC, returns COM_OK com_err_t
*/
com_err_t com_parse_packet(const uint8_t *buf, size_t buf_len, com_packet_t *pkt);

/* Write up to COM_MAX_PAYLOAD bytes of data to data BRAM at addr.
 * Splits automatically into multiple packets if data_len > COM_MAX_PAYLOAD. */
com_err_t com_write_data(com_dev_t *dev, uint16_t addr,
                           const uint8_t *data, size_t data_len);

/* Read data_len bytes from data BRAM starting at addr into out_buf. */
com_err_t com_read_data(com_dev_t *dev, uint16_t addr,
                          uint8_t *out_buf, size_t data_len);

#ifdef __cplusplus
}
#endif

#endif
