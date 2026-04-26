#ifndef UART_CONSTANTS
#define UART_CONSTANTS
#define UART_SOF            0xAAu
#define UART_MAX_PAYLOAD    255u
#define UART_HEADER_SIZE    6      /* SOF + CMD + ADDR_HI + ADDR_LO + LEN  */
#define UART_OVERHEAD       7      
#define UART_MAX_PACKET     (UART_OVERHEAD + UART_MAX_PAYLOAD)



#endif
