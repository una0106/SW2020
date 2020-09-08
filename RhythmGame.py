import cv2
import time
from utils import *
from statistics import median_high
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path
import tf_pose.common as common
import pygame

score = 0

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def shownum(named_window, target_time, play_time, image):       #포즈에서 사용 => 필요 없음
    if target_time - 3 <= play_time <= target_time :
        cv2.imshow('McgBcg', cv2.imread(image))
    elif play_time > target_time - 3.5:
        cv2.putText(named_window, '1', (config.imWidth - 555, config.imHeight - 160), cv2.FONT_HERSHEY_TRIPLEX, 4, (0, 0, 255), 7, cv2.LINE_8) # 1일 때 빨간색
    elif play_time > target_time - 4.5:
        cv2.putText(named_window, '2', (config.imWidth - 555, config.imHeight - 160), cv2.FONT_HERSHEY_TRIPLEX, 4, (255, 255, 255), 7, cv2.LINE_8)
    elif play_time > target_time - 6.5:
        cv2.putText(named_window, '3', (config.imWidth - 555, config.imHeight - 160), cv2.FONT_HERSHEY_TRIPLEX, 4, (255, 255, 255), 7, cv2.LINE_8)

def show_hp(bgImg, hp_img, x_offset, y_offset, x_resize, y_resize):
    hp_img = cv2.resize(hp_img, (x_resize, y_resize))
    rows, cols, channels = hp_img.shape
    roi = bgImg[y_offset: rows + y_offset, x_offset: x_offset + cols]
    #검정 -> 흰색으로 색변환하고 바탕 투명하게 하는 -> 합성함
    img2gray = cv2.cvtColor(hp_img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    bgImg_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    hp_fg = cv2.bitwise_and(hp_img, hp_img, mask=mask)
    dst = cv2.add(bgImg_bg, hp_fg)
    bgImg[y_offset: y_offset + rows, x_offset:x_offset + cols] = dst

def match(config, match_list, centers, hp, play_time):
    BodyColors = [[255, 0, 0],  #각 부위별 색깔 다르게 함 -> 게임할때 원에 같은 색의 부위가 오게
                  [0, 0, 0],
                  [0, 0, 0],
                  [255, 102, 0],
                  [255, 255, 0],
                  [0, 0, 0],
                  [255, 102, 0],
                  [255, 255, 0],
                  [0, 153, 0],
                  [0, 0, 0],
                  [0, 0, 255],
                  [0, 153, 0],
                  [0, 0, 0],
                  [0, 0, 255],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0]]
    for i in match_list:  # 예)i = [4.0, 3.5, 4.2, F, 0 or PATH, (2, 3), (5, 12)] # 여기 ~ 33 !!
        # if not i[4] == 0:
        #     pass

        # 위에 for문 화면에 bodypoint별로 원 그려주기
        for j in range(18):
            center = (int(centers[j][0]), int(centers[j][1]))
            color = [BodyColors[j][2],BodyColors[j][1],BodyColors[j][0]]        #우리는 rgb로 넣었는데 opencv는 gbr
            config.named_window = cv2.circle(config.named_window,
                                             center, 10, color, thickness=-1)

        for j in i[5:]:  # 5 인덱스부터 끝까지 예)j = (2, 3) #구역번호, 부위번호
            if i[0] - 3 < play_time < i[0]:
                circle_ratio = (play_time - (i[0] - 3)) / 3  # 3.7 ~ 최대 4.0초
                box_x = int((config.activation_areas[j[0]][0][0] + config.activation_areas[j[0]][1][0]) / 2)
                box_y = int((config.activation_areas[j[0]][0][1] + config.activation_areas[j[0]][1][1]) / 2)

                color = [BodyColors[j[1]][2], BodyColors[j[1]][1], BodyColors[j[1]][0]]

                config.named_window = cv2.circle(config.named_window,   #원 그려줌
                           (box_x, box_y), 20,
                           color, thickness=-1)
                config.named_window = cv2.circle(config.named_window,
                           (box_x, box_y),
                           60 - int(40 * circle_ratio), color, thickness=2)
            
            #사용자의 부위 좌표가 저장된 centers 리스트가 각각 적절한 구역의 좌표 범위 안에 있는지 확인
            #x좌표 사이에 centers오게 / y좌표 사이에 centers오게
            #j[0]은 구역번호
            if int(config.activation_areas[j[0]][0][0]) < centers[j[1]][0] < int(config.activation_areas[j[0]][1][0]) and int(config.activation_areas[j[0]][0][1]) < centers[j[1]][1] < int(config.activation_areas[j[0]][1][1]): # ?? 범
                  # and i[4] == False: 지움 !!
                global score
                score += 5      #들어오면 점수 올려줌
                if hp < 10: #맞추면 목숨 2개 올려주고
                    hp += 2
                if hp > 10:#목숨 최대 10개
                    hp = 10

                match_list.remove(i)        #한번 검사하면 match list에서 빼주기

    return match_list  # global화 시키기 위해서 return


def start_game(config, params):
    #게임 들어가기 전 필요한 변수들 초기화
    cam = cv2.VideoCapture(0)
    ret, named_window = cam.read()


    # 실루엣 맞추기: 카메라 키고, (사진 띄우고, point 4개 범위 안에 들어오면) X 3번 loop 나가
    # sil = ["1.png", "2.png", "3.png"] # 이런 식

    # 게임 시작: clear_menu, pause_menu, death_menu 중에 하나로 끝남
    pause_img = cv2.imread('images/pause.png')
    score_img = cv2.imread('images/score.png')
    gameover_img = cv2.imread('images/gameover.png')

    # 목숨 관련 변수들
    hp_x = config.imWidth//2 + 400
    hp_y = config.imHeight//2 - 345
    hp_yy = config.imHeight//2 - 300
    hp_w = 50
    hp_h = 42
    hp_image = cv2.imread('images/heart.png')

    w = 432
    h = 368
    e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(w, h), trt_bool=str2bool("False"))

    global score
    while True:  # restart 하면 여기로 돌아오지 (실루엣 다시 안 해도 됨)
        params["restart"] = False
        hp = 10 # death까지의 목숨(?) (10번 못 맞추면 death_menu)
        cur_order = 0
        # params

        score = 0

        game_patterns = [] # 재구성할 리스트
        
        #엑셀에서 불러 온 값
        for i in params["patterns"]: # ex) i = [4.,0 0, 0, 3, 0, 0, 12, 0, 0, 0] 여기 ~ 89 !!
            list = []
            if i[10]:           #포즈를 위해서 i[10]이 true면 포즈 있는거여서 포즈 취할 시간줌 => 필요없음
                time1 = i[0] - 6.6
                time2 = i[0]
            else:                   #포즈 없는 경우 -> 원에 사람의 bodypoint touch할 시간의 범위를 줌
                time1 = i[0] - 3 # 여기 ~ 81!!
                time2 = i[0] + 1
            list.extend([i[0], time1, time2, False, i[10]])     #원래는i[0]시간인데 time1~time2시간의 범위를 주겠다
            # 구역 9개에 대해서 리스트에다가 (영역, 부위) 튜플을 원소로 append
            for j in range(1, 10): # j = 1 ~ 9
                if i[j]:        #0이 아니면...원이 나와야됨
                    list.append(tuple([j - 1, i[j] - 1]))   #excel에서 초시간때문에 구역 번호랑 -1차이 -> j-1
            game_patterns.append(list)                      #i[j]-1 : excel에 잘못 적음->일일이 고치기 귀찮아서 -> i[j]-1

        # params["patterns"][0] = [4,0, 0, 0, 3, 0, 0, 12, 0, 0, 0]
        #   -> game_patterns[0] = [4.0, 3.5, 4.2, False, (2, 2), (5, 11)]  (구역번호, 부위번호)
        match_list = [] # 주어진 시간 안에 해당되는, match 해볼 규칙들

        #a = input('Press...')

        start_time = time.time()
        resume_time = 0.0
        resume_start = 0.0
        play_music(params["song"], 0)
        while True: # game play

            ret, named_window = cam.read()
            config.named_window = cv2.resize(named_window, dsize=(1312, 736), interpolation=cv2.INTER_AREA)
            config.named_window = cv2.flip(config.named_window, 1)
            print(named_window.shape)
            humans = e.inference(named_window, resize_to_default=(w > 0 and h > 0), upsample_size=4.0) # 4 / 1 ??
            if not humans:
                continue

            human = humans[0]

            image_h, image_w = config.named_window.shape[:2]

            #Human 클래스의 add_pair 함수(estimator.py의 62줄)로 포인트를 파악하고, 파악한 좌표를 centers 리스트에 저장
            #->머리부터 발끝까지의 키 포인트들이 화면에 표시됩니다.
            centers = []
            for i in range(common.CocoPart.Background.value):       #18번
                if i not in human.body_parts.keys():
                    centers.append((0, 0))
                else:
                    body_part = human.body_parts[i]
                    center = (image_w - int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
                    centers.append(center)          #사람의 keypoint받아서 화면에 출력

            # 실루엣
            play_time = time.time() - start_time  # 플레이 시간 측정
            pattern = game_patterns[cur_order]

            # 어떤 규칙이 time1을 지나면 & 아직 match_list에 없으면(= 첫번째 조건 만족해도 중복 append 방지 위해)
            #game_patterns[cur_order][1]는 맞춰야 하는 시간 범위의 최솟값 && match_list에 없으면....
            if game_patterns[cur_order][1] < play_time and game_patterns[cur_order] not in match_list:
                match_list.append(game_patterns[cur_order])
                # cur_pattern = Pattern()
                cur_order += 1
                if cur_order > len(game_patterns) - 1:      #이 조건을 만족하면 게임이 끝난것 ->cur_order고정 -> game 종료
                    cur_order = len(game_patterns) - 1
            if match_list:      #matchlist에 원소가 하나라도 있으면 아래 인자들 match함수에 넘겨줌
                # centers resize, flip      i = [4.0, 3.5, 4.2, F, 0 or PATH, (2, 3), (5, 12)] # 여기 ~ 33 !
                match_list = match(config, match_list, centers, hp, play_time)      #=> 위에 match 함수 가기~!!!!
            if match_list and match_list[0][2] < play_time: # and 아직 있으면        #터치해야 할 시간 지났음 -> 목숨 하나 빼기
                hp -= 1
                del match_list[0] # 고침!! 항상 [0]일 테니끼 right?     #끝나면 match_list에서 지우니까 항상 [0]지움
                # match_list.remove(game_patterns[cur_order]) 도 됨

            cv2.putText(config.named_window, 'score:', (int(config.imWidth / 2 - 600), int(config.imHeight / 2 - 300)), cv2.FONT_HERSHEY_PLAIN, 4,
                        (255, 255, 255), 7, cv2.LINE_8) #실시간으로 점수 보여주기
            cv2.putText(config.named_window, '%d' % score, (int(config.imWidth / 2 - 600), int(config.imHeight / 2 - 250)), cv2.FONT_HERSHEY_PLAIN, 4,
                        (255, 255, 255), 7, cv2.LINE_8)

            if cur_order == len(game_patterns): # 이런 식      #게임이 끝났으면(재구성한 list가) -> clear_menu보여주기
                config.named_window = score_img
                clear_menu(params, score)

            if cv2.waitKey(1) & 0xFF == ord('p'):
                params["exit"] = True

            if hp <= 0 or play_time > game_patterns[len(game_patterns) - 1][2] + 5:
                                #마지막 game_patterns의 터치 허용 범위 시간이 지나고도 5초뒤
                mixer.music.stop()
                death_menu(params)      #죽음


            if params["exit"] == True:
                break
            if params["restart"] == True: # 같은 게임 다시 시작
                break
            if params["menu"] == True:
                break

            for i in range(hp):
                if i < 5:       #실시간으로 변하는 window에 hp합성
                    show_hp(config.named_window, hp_image, hp_x + i * hp_w, hp_y, hp_w, hp_h)
                if i >= 5:      #2줄로 만들었음
                    show_hp(config.named_window, hp_image, hp_x + (i - 5) * hp_w, hp_yy, hp_w, hp_h)

            cv2.imshow('McgBcg', config.named_window) #image_h, image_w

        if params["exit"] == True:
            break
        if params["menu"] == True:
            break


def clear_menu(params, score): # 게임 잘 끝냈을 때

    play_sound(sound_applause)
    # show score
    cv2.putText(config.named_window, '%d' % score, (int(config.imWidth / 2 - 390), int(config.imHeight / 2 + 90)), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 7, (0,0,0), 15, cv2.LINE_8)
    cv2.putText(config.named_window, '%d'%score, (200, 480), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 7, (255,255,255), 15, cv2.LINE_8)
    cv2.imshow('McgBcg!', config.named_window)

    a = cv2.waitKey(0)
    while True:
        if a & 0xFF == ord('1'):
            play_sound(sound_effect2)
            params["menu"] = True
            print("menu")
            break
        if a & 0xFF == ord('1'):
            play_sound(sound_effect2)
            params["restart"] = True
            print("restart")
            break
        if a & 0xFF == ord('1'):
            play_sound(sound_effect2)
            params["exit"] = True
            print("exit")
            break


def death_menu(params): # 너무 못해서 알아서 게임이 멈춤
    play_sound(sound_disappointed)
    image = cv2.imread('images/gameover.png')
    while True:
        a = cv2.waitKey(1)
        cv2.imshow('McgBcg', image)
        if a & 0xFF == ord('1'): # restart
            play_sound(sound_effect2)
            print('restart')
            params["restart"] = True
            break
        if a & 0xFF == ord('2'): # menu
            play_sound(sound_effect2)
            print('menu')
            params["menu"] = True
            break
        if a & 0xFF == ord('3'): # exit
            play_sound(sound_effect2)
            print('exit')
            params["exit"] = True
            break
    cv2.destroyAllWindows()