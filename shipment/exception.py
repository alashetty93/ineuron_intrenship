import sys

class ShipmentException(Exception):
    
    def __init__(self,error_message,error_details:sys):
        
        self.error_message = error_message_details(
            error=error_message,
            error_details=error_details
        )

    def __str__(self):
        return self.error_message

#defined a function to get detailed error message
def error_message_details(error,error_details:sys):
    _,_,exc_tb  = error_details.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename

    error_message = f"Error occured python script name [{file_name} line no. [{exc_tb.tb_lineno}] error_message [{str(error)}]"

    return error_message