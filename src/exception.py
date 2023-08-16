
import sys
import logging

def error_message_details(error,error_detail:sys):
    _,_,exc_tb= error_detail.exc_info()           #--> it will give all the error message
    file_name= exc_tb.tb_frame.f_code.co_filename   #--> will return file name of error
    line_no = exc_tb.tb_lineno
    
    error_message = "Error occured in python script name [{0}] and line no. [{1}], error name [{2}]".format(
        file_name,line_no,str(error))
    
    return error_message



class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message,error_detail=error_detail)
        
    def __str__(self) -> str:     #--> it will print raised error message
        return self.error_message
        
    
try:
    a=1/0

except Exception as e:
    logging.info('Devide by zero')
    raise CustomException(e,sys)


    
    
        