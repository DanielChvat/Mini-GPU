#ifndef UART_CONSTANTS
#define UART_CONSTANTS
#define SIMT_SOF            0xAAu
#define SIMT_MAX_PAYLOAD    65536
#define SIMT_HEADER_SIZE    6      /* SOF + CMD + ADDR_HI + ADDR_LO + LEN  */
#define SIMT_OVERHEAD       7      
#define SIMT_MAX_PACKET     (SIMT_OVERHEAD + SIMT_MAX_PAYLOAD)



#endif