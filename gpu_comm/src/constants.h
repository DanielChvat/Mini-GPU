#ifndef COM_CONSTANTS
#define COM_CONSTANTS
#define COM_SOF            0xAAu
#define COM_MAX_PAYLOAD    255u
#define COM_HEADER_SIZE    6      /* SOF + CMD + ADDR_HI + ADDR_LO + LEN  */
#define COM_OVERHEAD       7      
#define COM_MAX_PACKET     (COM_OVERHEAD + COM_MAX_PAYLOAD)



#endif
