#Thư viện
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

#dữ liệu
data = pd.read_csv("USERS_DATA.csv", header=None).values
dd = pd.read_csv("TRAVEL_LOCATION.csv", header=None).values

# Tách các đặc trưng và nhãn để tiến hành huấn luyện
X = data[:,0:6]
y = data[:,-1]

@st.cache_data
def modelsRandomForest(X,y): # Xây dự mô hình Random Forest

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  # Tạo một đối tượng mô hình Random Forests
  random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)

  # Huấn luyện mô hình trên tập huấn luyện
  random_forest_model.fit(X_train, y_train)
  return random_forest_model

def Label(models, new_data): # hàm trả về giá trị dự đoán nhãn của models
  return models.predict(new_data)

def filter(data,new_data,label): # Hàm lọc sau khi đã có kết quả dự đoán
  tuoi = new_data[0][0]
  thunhap = new_data[0][2]
  noio = new_data[0][3]
  ngaynghi = new_data[0][4]

  fil1 = data[data[:,-1] == label]
  # Lọc lần 1 để lấy điểm dữ liệu trùng với nhãn

  fil2 = fil1[(fil1[:,0] > tuoi - 5)&(fil1[:,0] < tuoi + 5)]
  # Lọc lần 2 để lấy các điểm dữ liệu gần với tuổi của điểm dữ liệu mới

  fil3 = fil2[(fil2[:,2] > thunhap - 400)&(fil2[:,2] < thunhap + 400)]
  # Lọc lần 3 để lấy các điểm giá trị có mức thu nhập gần với điểm dữ liệu mới

  if (ngaynghi < 4):    # Nếu ngày nghỉ ít hơn 4 thì tiến hành lọc lần 4 để lấy các điểm dữ liệu ở gần nơi ở
    fil4 = fil3[(fil3[:,-2] > noio - 2)&(fil3[:,-2] < noio + 2)]
    diadiem,lan = np.unique(fil4[:,-3], return_counts=True) # Xem số lần xuất hiện của các địa điểm
    sort = np.argsort(-lan) # sắp xếp theo thứ tự giảm dần
    top3 = diadiem[sort][:3]
    return top3
  else: # Số lượng ngày nghỉ lớn hơn 4 thì sẽ lấy lần lọc thứ 3 để xác định địa điểm
    diadiem,lan = np.unique(fil3[:,-3], return_counts=True) # Xem số lần xuất hiện của các địa điểm
    sort = np.argsort(-lan) # sắp xếp theo thứ tự giảm dần
    top3 = diadiem[sort][:3]
    return top3
  
models = modelsRandomForest(X,y)

#joblib.dump(models, 'models.pkl')

#models = joblib.load('models.pkl')

st.title("Hệ thống đề xuất địa điểm du lịch tại Việt Nam")
# Tạo các trường nhập liệu
tuoi = st.number_input("Tuổi của bạn:", step=1, value=None)
if tuoi is not None and tuoi <= 0:
   st.warning("Vui lập nhập đúng tuổi của bạn!")
   st.stop()
gioitinh = st.selectbox("Giới tính: ", ["-Chọn-","Nữ", "Nam"])
if gioitinh == "Nữ":
    gioi_tinh_so = 0
elif gioitinh == "Nam":
    gioi_tinh_so = 1
elif gioitinh == "-Chọn-":
    gioi_tinh_so = None
thunhap = st.number_input("Thu nhập của bạn (VND):", step=1, value=None)
if thunhap is not None and thunhap > 0:
    thunhapvnd = thunhap/24405
elif thunhap is not None and thunhap <= 0:
    st.warning("Vui lòng nhập đúng thu nhập của bạn!")
    st.stop()
tinh_dict = {"-Chọn-" : None, "An Giang" : 54, "Bà Rịa - Vũng Tàu" : 49, "Bắc Giang" : 8, "Bắc Kạn" : 3, "Bạc Liêu" : 61, "Bắc Ninh" : 16, "Bến Tre" : 55, "Bình Định" : 35, "Bình Dương" : 46, "Bình Phước" : 45, "Bình Thuận" : 39, "Cà Mau" : 62, "Cần Thơ" : 63, "Cao Bằng" : 2, "Đà Nẵng" : 32, "Đắk Lắk" : 42, "Đắk Nông" : 43, "Điện Biên" : 12, "Đồng Nai" : 47, "Đồng Tháp" : 52, "Gia Lai" : 41, "Hà Giang" : 1, "Hà Nam" : 17, "Hà Nội" : 18, "Hà Tĩnh" : 28, "Hải Dương" : 19, "Hải Phòng" : 20, "Hậu Giang" : 58, "Hoà Bình" : 13, "Hưng Yên" : 21, "Khánh Hoà" : 37, "Kiên Giang" : 59, "Kon Tum" : 40, "Lai Châu" : 14, "Lâm Đồng" : 44, "Lạng Sơn" : 4, "Lào Cai" : 10, "Long An" : 51, "Nam Định" : 22, "Nghệ An" : 27, "Ninh Bình" : 23, "Ninh Thuận" : 38, "Phú Thọ" : 7, "Phú Yên" : 36, "Quảng Bình" : 29, "Quảng Nam" : 33, "Quảng Ngãi" : 34, "Quảng Ninh" : 9, "Quảng Trị" : 30, "Sóc Trăng" : 60, "Sơn La" : 15, "Tây Ninh" : 48, "Thái Bình" : 24, "Thái Nguyên" : 6, "Thanh Hoá"  : 26, "Thừa Thiên Huế" : 31, "Tiền Giang" : 53, "TP Hồ Chí Minh" : 50, "Trà Vinh" : 57, "Tuyên Quang" : 5, "Vĩnh Long" : 56, "Vĩnh Phúc" : 25, "Yên Bái" : 11}
chon_tinh = st.selectbox("Chọn nơi bạn sinh sống: ", list(tinh_dict.keys()))
tinh_id = tinh_dict[chon_tinh]
ngaynghi = st.number_input("Số ngày nghỉ:", step=1 ,value=None)
if ngaynghi is not None and ngaynghi <= 0:
   st.warning("Vui lòng nhập đúng số ngày bạn muốn nghỉ dưỡng!")
   st.stop()
chiphi = st.number_input("Chi phí của bạn (VND):", step=1, value=None)
if chiphi is not None and chiphi > 0:
    chiphivnd = chiphi / 24405
elif chiphi is not None and chiphi <= 0:
    st.warning("Vui lòng nhập đúng chi phí của bạn!")
    st.stop(  )
    


if st.button("Đề xuất"):
    if tuoi is None or gioi_tinh_so is None or thunhap is None or tinh_id is None or chiphi is None:
        st.warning("Vui lòng nhập đầy đủ thông tin!")
        st.stop()
    else:
      new_data = [[tuoi, gioi_tinh_so, thunhapvnd, tinh_id, ngaynghi, chiphivnd]]
#      st.write(label)
        #Dự đoán nhãn cho điểm dữ liệu mới
      label = Label(models,new_data)
      top3 = filter(data,new_data,label) # lấy ra 1 - 3 điểm xuất hiện nhiều nhất
      if len(top3) == 0:
       st.write("Xin lỗi, nhưng tôi không tìm thấy địa điểm nào phù hợp dành cho bạn!")
      else:
       st.write("Đây là ",len(top3)," địa điểm bạn nên đi du lịch:")
       for i in range(top3.shape[0]):
         for j in range(dd.shape[0]):
            if top3[i] == dd[j, 0]:
                st.write('<a href="'+"https://www.google.com/search?q=" + dd[j, 1] + " ở tỉnh " + dd[j, -1] +'" >'+dd[j, 1] + " ở tỉnh " + dd[j, -1] +'</a>', unsafe_allow_html=True)


