import psycopg2
import json
from util.json_decoder import Decoder
def insert_user(cursor,name,background_info):
  try:
    logger.debug("Inserting into user_info... {},{}".format(name,background_info))
    cursor.execute("INSERT INTO user_info (user_id,user_background_information) VALUES (%s, %s) ",(name,background_info))
  except (Exception, psycopg2.DatabaseError) as error:
    logger.error(error)

def insert_user_results(cursor,params,accuracy,time_taken,logger,user_id):
  try:
    logger.debug("Inserting into user_info... {},{},{}".format(user_id,params,accuracy))
    query = """ INSERT INTO user_results ( params, accuracy,time_taken,user_id) VALUES (%s,%s,%s,%s) RETURNING id"""
    record_to_insert = (params, accuracy,time_taken,user_id)
    cursor.execute(query, record_to_insert)
    id_row = cursor.fetchone()[0]
    logger.debug("Inserted row here {}".format(id_row))
    return id_row
    # rows = cursor.fetchall()
    # print(rows)
    # for row in rows:
    #   print(row)
    #cursor.execute("INSERT INTO variance_results (params, accuracy) VALUES (%s, %s) ",(params,accuracy))
  except Exception as error:
    logger.error(error)
    return None

def update_user_results(cursor,accuracy,time_taken,logger,row_id):
  try:
    logger.debug("updating user_info... {},{},{}".format(row_id,time_taken,accuracy))
    query = """ UPDATE user_results
                SET accuracy = %s, time_taken = %s
                WHERE id = %s"""
    record_to_update = (accuracy,time_taken,row_id)
    cursor.execute(query, record_to_update)
    logger.debug("Row updates successfully")
    return True
  except Exception as error:
    logger.error(error)
    return False

def insert_user_results_final(cursor,params,logger,user_id):
  try:
    logger.debug("Inserting into user_info... {},{}".format(user_id,params))
    query = """ INSERT INTO user_final_submissions ( params,user_id) VALUES (%s,%s)"""
    record_to_insert = (params, user_id)
    cursor.execute(query, record_to_insert)
    return True
  except Exception as error:
    logger.error(error)
    return False

def get_result_data(cursor,user_id,logger):
  try:
    logger.info("fetching results for user {}".format(user_id))
    query = """ SELECT params,accuracy FROM user_results WHERE user_id = %s"""
    cursor.execute(query, (user_id,))
    rows = cursor.fetchall()
    all_rows = []
    index=1
    if rows is not None:
      for row in rows:
        result = {}
        result["params"] = json.loads(row[0],cls=Decoder)
        result["accuracy"] = row[1]
        result["index"] = index
        index=index+1
        all_rows.append(result)
      return all_rows
    else:
      logger.debug("No results found for userid:{}".format(user_id))
      return None
  except Exception as error:
    logger.critical(error)
    return None

def fetch_all_users(cursor,logger):
  try:
    logger.info("fetching all users... ")
    query = """SELECT user_id FROM user_info"""
    cursor.execute(query)
    return cursor.fetchall()
    # rows = cursor.fetchall()
    # print(rows)
    # for row in rows:
    #   print(row)
    #cursor.execute("INSERT INTO variance_results (params, accuracy) VALUES (%s, %s) ",(params,accuracy))
  except Exception as error:
    logger.error(error)
    return None

def insert_background_info(cursor,data,logger):
  try:
    logger.info("Inserting into user_info id,experience,education... {},{},{}".format(data['user_id'],data['experience'],data['education']))
    query = """ INSERT INTO user_info ( user_id, user_experience,user_education_background) VALUES (%s,%s,%s)"""
    record_to_insert = (data['user_id'],data['experience'],data['education'])
    cursor.execute(query, record_to_insert)
    return {'status':1,'user_id':data['user_id'],'experience':data['experience'],'education':data['education']}
    # rows = cursor.fetchall()
    # print(rows)
    # for row in rows:
    #   print(row)
    #cursor.execute("INSERT INTO variance_results (params, accuracy) VALUES (%s, %s) ",(params,accuracy))
  except Exception as error:
    logger.error(error)
    return {'status':0,'error':str(error)}

def insert_precision(cursor,params,accuracy,time_taken,logger):
  try:
    logger.info("Inserting into variance_results... {},{}".format(params,accuracy))
    query = """ INSERT INTO variance_results ( params, accuracy,time_taken) VALUES (%s,%s,%s)"""
    record_to_insert = (params, accuracy,time_taken)
    cursor.execute(query, record_to_insert)
    return True
    # rows = cursor.fetchall()
    # print(rows)
    # for row in rows:
    #   print(row)
    #cursor.execute("INSERT INTO variance_results (params, accuracy) VALUES (%s, %s) ",(params,accuracy))
  except Exception as error:
    logger.error(error)
    return False
