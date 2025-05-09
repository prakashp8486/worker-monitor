a
    *�h�O  �                
   @   s�  d dl Z de _d dlZd dlZd dlZd dlZd dlZd dlm	Z	 z�e	� Z
e
d Ze
d Ze
d Zee
d �Zee
d �Ze
d	 d
kZee
d �Ze
d Ze
d Ze
d Ze
d Ze
d Ze
d Zee
d �Zee
d �Ze
d Zee
d �Zee
d �Zee
d �Zee
d �Z e
d d
kZ!g d�Z"W n> e#�yf Z$ z$e%de$� �� e �&d� W Y dZ$[$n
dZ$[$0 0 z$ej'ej(de�)e�e�*� gd� W n4 e#�y� Z$ ze%de$� �� W Y dZ$[$n
dZ$[$0 0 zd d l+m,Z, W nH e-�y Z$ z.e�.d!e$� �� e%d"� e �&d� W Y dZ$[$n
dZ$[$0 0 g a/d#a0d$d%� Z1d5d&d'�Z2d(d)� Z3d*d+� Z4d,d-� Z5d.d/� Z6d0d1� Z7e8d2k�r�z
e7�  W nZ e9�y�   e�:d3� Y n> e#�y� Z$ z$e�;d4e$� �� e�<�  W Y dZ$[$n
dZ$[$0 0 dS )6�    NT)�config_reader�project_nameZauthor�mode�
image_size�classes�persist�True�conf_threshold�log_file_pathZpretrained_model_pt_pathZpretrained_model_onnx_path�	rtsp_linkZinference_video�
roi_points�verbose_history�iou_threshold�	dis_lines�idle_alert_threshold�movement_threshold�resize_width�resize_height�normalize_frames))�d   r   )�  r   )r   �|  )r   r   zError reading config: �   z)%(asctime)s - %(levelname)s - %(message)s)�level�format�handlersz&Warning: Could not configure logging: )�YOLOzFailed to import YOLO: z3Please install ultralytics: pip install ultralyticsFc              
   C   s�   z�| \}}t |�}d}|d \}}t|d �D ]�}|||  \}	}
|t||
�kr�|t||
�kr�|t||	�kr�||
kr�|| |	|  |
|  | }||	ks�||kr�| }|	|
 }}q.|W S  ty� } zt�d|� �� W Y d }~dS d }~0 0 d S )NFr   r   zError in point_in_polygon: )�len�range�min�max�	Exception�logging�error)ZpointZpolygon�x�y�nZinsideZp1xZp1y�iZp2xZp2yZx_intersect�e� r)   �main.py�point_in_polygon@   s&    r+   c                 C   sZ  | d u rt } z�g }t| d���}|D ]p}z,|�� �d�\}}|�t|�t|�f� W q" ty� } z&t�d|�� � d|� �� W Y d }~q"d }~0 0 q"W d   � n1 s�0    Y  t	|�dkr�t�
d| � �� |W S t�d| � dt	|�� �� W d S W n^ t�y   t�d	| � d
�� Y d S  t�yT } zt�d|� �� W Y d }~d S d }~0 0 d S )N�r�,zInvalid line in ROI file: z	, Error: �   zROI points loaded from zInvalid number of points in z, expected 4, got z	ROI file z
 not foundzCould not load ROI points: )�roi_file_path�open�strip�split�append�int�
ValueErrorr"   r#   r   �info�warning�FileNotFoundErrorr!   )�filenameZpoints�f�liner$   r%   r(   r)   r)   r*   �load_roi_from_fileW   s,    P
r<   c              
   C   s�   zvt |�dk r4t�d� tj| jd d� tjd�W S tj| jd d� tjd�}t�|tj	��
d�}t�||gd� |W S  ty� } z6t�d|� �� tj| jd d� tjd�W  Y d }~S d }~0 0 d S )N�   z)Not enough ROI points to create a polygon�   )Zdtype������r   r>   ��   zError creating mask: )r   r"   r#   �npZones�shape�uint8Zzeros�array�int32�reshape�cv2ZfillPolyr!   )�framer   �mask�roi_polygonr(   r)   r)   r*   �create_maskq   s    
rL   c              
   C   sT   zt j| | |d�}|W S  tyN } z t�d|� �� | W  Y d }~S d }~0 0 d S )N)rJ   zError extracting ROI: )rH   Zbitwise_andr!   r"   r#   )rI   rJ   Zroir(   r)   r)   r*   �extract_roi   s    rM   c              
   C   sV   zt j| ||ft jd�W S  tyP } z t�d|� �� | W  Y d}~S d}~0 0 dS )z-Resize the frame to the specified dimensions.)ZinterpolationzError resizing frame: N)rH   ZresizeZ
INTER_AREAr!   r"   r#   )rI   �widthZheightr(   r)   r)   r*   �resize_frame�   s
    rO   c              
   C   sP   z| � tj�d W S  tyJ } z t�d|� �� | W  Y d}~S d}~0 0 dS )z'Normalize pixel values to range [0, 1].g     �o@zError normalizing frame: N)�astyperB   Zfloat32r!   r"   r#   )rI   r(   r)   r)   r*   �normalize_frame�   s
    rQ   c            /         s|  �z6t t�} | r| adantat�d� zht�� dkrRt	�
t�}t�dt� �� n<t�� dkrzt	�
t�}t�dt� �� nt	�
d�}t�d� W nN tttfy� } z0t�d	|� �� t	�
d�}t�d
� W Y d }~n
d }~0 0 |�� s�t�d� W d S zJzttdd�}t�d� W n* t�y>   ttdd�}t�d� Y n0 W n: t�y| } z t�d|� �� W Y d }~W d S d }~0 0 i }i }t�� }t�dt� dt� �� t�dt� �� �	z�|�� \}}|�s�t�d� W �q�z4|�� }	t|tt�}|�� }
t�rt|�}n|}W n> t�yL } z$t�d|� �� |}
|}W Y d }~n
d }~0 0 t|t�}t||�}t �� dk�r�z0t!�"tt!j#�}|�$d�}t	�%|
|gddd� W n6 t�y� } zt�d|� �� W Y d }~n
d }~0 0 t�r�|d �&t!j'�}n|}z|j(|t)t*t+t,t-d�}W n: t�yH } z t�d|� �� d }W Y d }~n
d }~0 0 t.� � d}d}|�r�t/|�dk�r�zB|d j0}t1|d��r�|j2d u�r�z$|j2�3� �4� �5� }|j6�4� �7� }W n> t�y� } z$t�d|� �� g }g }W Y d }~n
d }~0 0 t8||�D �]�\}}�zft9t3|�\}}}}|| d  }|| d  }||f}zR|dk �s�||j:d k�s�|dk �s�||j:d k�s�|||f dk�r�W W �q W n4 t;�y�   t�d!|� d"|j:� �� Y W �q Y n0 |d7 }� �<|� zt	�=|
|d#d$d%� W n6 t�y& } zt�d&|� �� W Y d }~n
d }~0 0 d}d} ||v �r�z6|| \}!}"t!�>||! d  ||" d   �} | t?k}W n6 t�y� } zt�d'|� �� W Y d }~n
d }~0 0 ||f||< z�t�� }#||v�r�|�s�|#nd |d(d)d*�||< nX|�rd || d+< d|| d,< d)|| d-< n,|�s>|| d, �r>|#|| d+< d)|| d,< W n6 t�yv } zt�d.|� �� W Y d }~n
d }~0 0 d(}$d)}%zH|| d, �s�|| d+ d u�r�|#|| d+  }$|$t@k}%|%|| d-< W n6 t�y� } zt�d/|� �� W Y d }~n
d }~0 0 |%�r|d7 }|%�rd0nd1}&zt	�A|
||f||f|&d � W n6 t�yp } zt�d2|� �� W Y d }~n
d }~0 0 |%�r|d3nd4}'d5|$d6�d7�}(zDt	�B|
|'||d8 ft	jCd9|&d � t	�B|
|(||d: ft	jCd9|&d � W n6 t�y } zt�d;|� �� W Y d }~n
d }~0 0 |%�rpd<})z$t	�B|
|)||d= ft	jCd>d0d � W n6 t�yn } zt�d?|� �� W Y d }~n
d }~0 0 W n6 t�y� } zt�d@|� �� W Y d }~n
d }~0 0 �q W n6 t�y� } zt�dA|� �� W Y d }~n
d }~0 0 z<� fdBdC�|�D� D �}*|*D ]}+|+|v �	r||+= ||+= �	qW n6 t�	yZ } zt�dD|� �� W Y d }~n
d }~0 0 z0t�� }#|#| },|,dk�	r�dE|, }-nd}-|#}W n: t�	y� } z t�dF|� �� d}-W Y d }~n
d }~0 0 z�t	�B|
dGtE� �dHt	jCdIdd� t	�B|
dJ|-d6��dKt	jCdIdd� t	�B|
dL|� �dMt	jCdIdd� t	�B|
dN|� �dOt	jCdId0d� t	�B|
dPt� dt� �dQt	jCdIdd� t	�B|
dRt�
rzdSndT� �dUt	jCdIdd� W n6 t�
y� } zt�dV|� �� W Y d }~n
d }~0 0 zt	�FtG|
� W nF t�y  } z,t�dW|� �� W Y d }~W �q�W Y d }~n
d }~0 0 z2t	�Hd�d@ }.|.tIdX�k�rRt�dY� W W �q�W n6 t�y� } zt�dZ|� �� W Y d }~n
d }~0 0 W nL t�y� } z2t�d[|� �� tJ�K�  W Y d }~�q�W Y d }~n
d }~0 0 �q�zt�d\� |�L�  t	�M�  W n6 t�y4 } zt�d]|� �� W Y d }~n
d }~0 0 W n> t�yv } z$t�Nd^|� �� tJ�K�  W Y d }~n
d }~0 0 d S )_NTzUsing default ROI pointsZvideozInferencing on video: �rtspzInferencing on RTSP stream: r   z'Using OpenCV's VideoCapture with webcamzError setting up video source: zFalling back to default webcamzError opening video captureZdetect)Ztaskz(YOLO model loaded successfully from ONNXz&YOLO model loaded successfully from PTzError loading YOLO model: zFrame resizing enabled: r$   zFrame normalization enabled: z Could not read frame. Exiting...zError processing frame: r%   r?   )rA   rA   rA   r   zError drawing ROI polygon: rA   )ZiouZimgszr   Zconfr   zError during YOLO inference: �idz Error extracting tracking data: r>   zCenter point z out of bounds for mask shape r.   )rA   r   rA   r@   zError drawing center point: zError calculating movement: g        F)�start_idle_time�	is_movingZtotal_idle_time�idle_statusrT   rU   rV   zError updating idle times: zError calculating idle status: )r   r   rA   )r   rA   r   zError drawing bounding box: ZIDLEZMovingzIdle time: z.1f�s�
   g      �?�   zError adding status text: zIDLE DETECTED!�2   g333333�?zError adding alert text: zError processing detection: z$Error processing detection results: c                    s   g | ]}|� vr|�qS r)   r)   )�.0�pid�Z
active_idsr)   r*   �
<listcomp>s  �    zmain.<locals>.<listcomp>z Error removing expired records: g      �?zError calculating FPS: zPREPARED BY: )rX   �   gffffff�?zFPS: )rX   �(   zPeople in ROI: )rX   �<   zIdle people: )rX   �P   zFrame size: )rX   r   zNorm: ZYesZNo)rX   �x   zError adding stats to frame: zError displaying frame: �qz User requested exit with 'q' keyzError checking for key press: zCritical error in main loop: zCleaning up resourceszError during cleanup: zFatal error in main function: )Or<   r/   r   �roi_selected�default_roi_pointsr"   r7   r   �lowerrH   ZVideoCapture�
video_pathr6   r   �	NameError�AttributeErrorr!   r#   ZisOpenedr   �model_path_onnx�model_path_pt�timer   r   r   �read�copyrO   rQ   rL   rM   r   rB   rE   rF   rG   Z	polylinesrP   rD   Ztrackr   r   r   r	   r   �setr   �boxes�hasattrrS   r4   Zcpu�tolist�xyxy�numpy�zip�maprC   �
IndexError�addZcircleZsqrtr   r   Z	rectangleZputTextZFONT_HERSHEY_SIMPLEX�keys�author_nameZimshowr   ZwaitKey�ord�	traceback�	print_exc�releaseZdestroyAllWindows�critical)/Z
loaded_roiZcapr(   ZmodelZperson_idle_timesZlast_positionsZ
start_time�retrI   Zoriginal_frameZdisplay_frameZprocessed_framerJ   Z	roi_framerK   Zinference_frame�resultsZpeople_in_roiZidle_people_countrr   Z	track_idsru   Ztrack_idZboxZx1Zy1Zx2Zy2Zcenter_xZcenter_yZcenter_pointrU   ZmovementZprev_xZprev_yZcurrent_timeZ	idle_timerV   ZcolorZstatus_textZ	time_textZ
alert_textZexpired_person_idsr\   Z	time_diffZfps�keyr)   r]   r*   �main�   s�   




 





&
F
&
&
�& &&
�
�&
�*.&
&

 $,&&
*(
*r�   �__main__z(Program terminated by keyboard interruptzUnhandled exception: )N)=�sys�dont_write_bytecoderH   rn   rv   rB   r~   r"   Zassets.config_readerr   �datar   r|   r   r4   r   r   r   �floatr	   r
   rm   rl   r   ri   r/   r   r   r   r   r   r   r   r   rg   r!   r(   �print�exitZbasicConfig�INFOZFileHandlerZStreamHandlerZultralyticsr   �ImportErrorr#   r   rf   r+   r<   rL   rM   rO   rQ   r�   �__name__�KeyboardInterruptr6   r�   r   r)   r)   r)   r*   �<module>   s�    ��
$ 
  

