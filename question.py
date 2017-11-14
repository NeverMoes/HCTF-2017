import sys
from django.conf import settings
from django.conf.urls import url
from django.http import JsonResponse
import hashlib
import pickle
import sqlite3 as sql
import base64
import gym

salt = 'NeverMoes-hctf'
token_api = ''

settings.configure(
    DEBUG=False,
    SECRET_KEY='thisisthesecretkey',
    ROOT_URLCONF=__name__
)

##############################################################################################

def game(request):
    id = request.GET.get('id')
    move = request.GET.get('move')
    status = requests.get(token_api+id).json()['status']
    if id and (move=='1' or move=='0') and status=='success':
        move = int(move)
        conn = sql.connect('hctf-ai.db')
        cursor = conn.cursor()
        row = cursor.execute('select * from hctf where id="%s"' % id).fetchone()
        if row:
            env = pickle.loads(base64.b64decode(row[1]))
            count = row[2]
            observation, _, done, _ = env.step(move)
            if done:
                cursor.execute('DELETE FROM hctf WHERE id="%s"' % (id,))
                conn.commit()
                conn.close()
                return JsonResponse({'count': count, 'status': False})
            else:
                if count >= 100:
                    cursor.execute('DELETE FROM hctf WHERE id="%s"' % id)
                    conn.commit()
                    conn.close()
                    return JsonResponse({'flag':'hctf{' + hashlib.sha256((id+salt).encode('utf-8')).hexdigest() + '}', 'status': False, 'count':100})
                else:
                    cursor.execute('UPDATE hctf SET env="%s", count=%d where id="%s"' % (base64.b64encode(pickle.dumps(env)).decode(), count+1, id))
                    conn.commit()
                    conn.close()
                    return JsonResponse({'observation': list(observation), "count": count, "status": True})
        else:
            env = gym.make('CartPole-v0')
            observation = env.reset()
            cursor.execute("insert into hctf(id, env, count) VALUES ('%s', '%s', %d)"
                           % (id, base64.b64encode(pickle.dumps(env)).decode(), 1))
            conn.commit()
            conn.close()
            return JsonResponse({"observation": list(observation), 'count': 1, "status": True})
    else:
        return JsonResponse({"status": False})

urlpatterns = (
    url(r'^game$', game),
)

###############################################################################################

if __name__ == "__main__":

    conn = sql.connect('hctf.db')
    cursor = conn.cursor()
    cursor.execute("""create table if NOT EXISTS hctf(
                      id text not null,
                      env text not null,
                      count int not null)
                   """)
    conn.commit()
    conn.close()

    from django.core.management import execute_from_command_line
    execute_from_command_line(sys.argv)

