import sys
from src.logger import logging





def err_message(error , err_det:sys):
    _,_,exc_tb = err_det.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Ehtisham error has occured in your execution , file  name is  '[{0}]' line number '[{1}]' error message '[{2}]'".format(
        file_name, exc_tb.tb_lineno,str(error))
        

    return error_message
    





class Custom_exception_handling(Exception):
    def __init__(self, error_message , error_details: sys):
        super().__init__(error_message)
        self.error_message = err_message(error_message , err_det=error_details)

    def __str__(self):
        return self.error_message





