from datetime import datetime
import logging

def mp_logger(logger_name):
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y_%m_%d_%H_%M_%S")
    save_dir = '/data3/xupin/0_UNName/logs/'+formatted_datetime+'_'+logger_name+'.log'
    
    logging.basicConfig(format='PPID: %(process)d - Time: %(asctime)s - %(levelname)s | %(message)s', 
                        level=logging.INFO, 
                        filename=save_dir.format('str'), 
                        filemode='a')
    logger = logging.getLogger(logger_name)
    return logger

def distinct_logger(logger_name):
    #! 设置时间
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y_%m_%d_%H_%M_%S")
    
    #!
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # 防止重复添加处理器
    if not logger.handlers:
        # 创建一个文件处理器
        fh = logging.FileHandler('/data3/xupin/0_UNName/logs/'+formatted_datetime+'_'+logger_name + '.log')
        fh.setLevel(logging.INFO)
        
        # 创建日志格式
        formatter = logging.Formatter('PPID: %(process)d - Time: %(asctime)s - %(levelname)s | %(message)s')
        fh.setFormatter(formatter)
        
        # 将处理器添加到日志记录器
        logger.addHandler(fh)
    
    return logger

if __name__ == '__main__':
    for edge_type_index in range(2):
        logger_name = 'test_'+str(edge_type_index+1)
        main_logger = distinct_logger(logger_name)
        main_logger.info('Augmenting Type ' + str(edge_type_index + 1) + ' Homo SubGraphList......')
