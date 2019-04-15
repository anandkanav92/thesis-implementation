import psycopg2
def insert_user(cursor,name,background_info):
  try:
    print("Inserting into user_info... {},{}".format(name,background_info))
    cursor.execute("INSERT INTO user_info (user_id,user_background_information) VALUES (%s, %s) ",(name,background_info))
  except (Exception, psycopg2.DatabaseError) as error:
    print(error)

def insert_precision(cursor,params,accuracy,time_taken):
  try:
    print("Inserting into user_info... {},{}".format(params,accuracy))
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
    print(error)
    return False
