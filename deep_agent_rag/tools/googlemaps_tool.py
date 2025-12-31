"""
Google Maps 工具
提供地點驗證、標準化、交通時間計算等功能
"""
import googlemaps
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple

from ..config import NORMAL_GOOGLE_MAPS_API_KEY, USER_HOME_ADDRESS, USER_OFFICE_ADDRESS


# 初始化 Google Maps 客戶端
_gmaps_client = None


def get_gmaps_client():
    """獲取 Google Maps 客戶端實例（單例模式）"""
    global _gmaps_client
    if _gmaps_client is None:
        if not NORMAL_GOOGLE_MAPS_API_KEY:
            raise ValueError("❌ Google Maps API Key 未設置，請在 .env 文件中設置 NORMAL_GOOGLE_MAPS_API_KEY")
        _gmaps_client = googlemaps.Client(key=NORMAL_GOOGLE_MAPS_API_KEY)
    return _gmaps_client


def validate_and_standardize_address(address: str, search_nearby: bool = True) -> Dict[str, any]:
    """
    驗證並標準化地址
    
    Args:
        address: 輸入的地址字符串
        search_nearby: 當地址模糊時，是否嘗試搜索附近地點（預設 True）
    
    Returns:
        包含以下欄位的字典：
        - success: bool - 是否成功
        - standardized_address: str - 標準化後的地址
        - coordinates: dict - 包含 'lat' 和 'lng' 的座標字典
        - place_id: str - Google Places ID
        - is_ambiguous: bool - 地址是否模糊（需要用戶確認）
        - suggestions: list - 建議的地點列表（如果地址模糊）
        - error: str - 錯誤訊息（如果失敗）
    """
    if not address or not address.strip():
        return {
            "success": False,
            "error": "地址為空"
        }
    
    try:
        gmaps = get_gmaps_client()
        
        # 檢測是否為模糊地址（包含"附近"、"周圍"、"附近的"等關鍵字）
        ambiguous_keywords = ["附近", "周圍", "附近的", "周邊", "around", "nearby", "near"]
        is_ambiguous_query = any(keyword in address for keyword in ambiguous_keywords)
        
        # 進行地理編碼
        geocode_result = gmaps.geocode(address)
        
        if not geocode_result:
            # 如果地址模糊且允許搜索附近，嘗試使用 Places API
            if is_ambiguous_query and search_nearby:
                return _search_nearby_places(address)
            
            return {
                "success": False,
                "error": f"找不到地址：{address}。請提供更具體的地點，例如：具體地址、地標名稱或餐廳名稱。",
                "is_ambiguous": is_ambiguous_query
            }
        
        # 解析結果
        result = geocode_result[0]
        location = result['geometry']['location']
        formatted_address = result['formatted_address']
        place_id = result.get('place_id', '')
        
        # 檢查結果是否準確（如果地址太模糊，可能返回錯誤的位置）
        if is_ambiguous_query:
            # 對於模糊地址，即使找到了結果，也標記為需要確認
            return {
                "success": True,
                "standardized_address": formatted_address,
                "coordinates": {
                    "lat": location['lat'],
                    "lng": location['lng']
                },
                "place_id": place_id,
                "original_address": address,
                "is_ambiguous": True,
                "suggestions": None
            }
        
        return {
            "success": True,
            "standardized_address": formatted_address,
            "coordinates": {
                "lat": location['lat'],
                "lng": location['lng']
            },
            "place_id": place_id,
            "original_address": address,
            "is_ambiguous": False,
            "suggestions": None
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"地址驗證失敗：{str(e)}"
        }


def _search_nearby_places(query: str, max_results: int = 5) -> Dict[str, any]:
    """
    使用 Places API 搜索附近的地點（當地址模糊時）
    
    Args:
        query: 搜索查詢（例如："附近的餐廳"）
        max_results: 最大返回結果數
    
    Returns:
        包含建議地點的字典
    """
    try:
        gmaps = get_gmaps_client()
        user_location = get_user_default_location()
        
        if not user_location:
            # 如果沒有預設位置，無法搜索附近地點
            return {
                "success": False,
                "error": f"地址「{query}」太模糊，無法找到確切位置。請提供具體地址或地標名稱。\n💡 提示：在 .env 文件中設置 USER_HOME_ADDRESS 或 USER_OFFICE_ADDRESS 可啟用附近地點搜索功能。",
                "is_ambiguous": True
            }
        
        # 先獲取用戶位置的座標
        user_geocode = gmaps.geocode(user_location)
        if not user_geocode:
            return {
                "success": False,
                "error": f"無法獲取您的預設位置座標，請檢查 USER_HOME_ADDRESS 或 USER_OFFICE_ADDRESS 設置。",
                "is_ambiguous": True
            }
        
        user_coords = user_geocode[0]['geometry']['location']
        
        # 提取地點類型（例如："附近的餐廳" -> "餐廳"）
        place_type = query
        for keyword in ["附近", "周圍", "附近的", "周邊", "around", "nearby", "near"]:
            place_type = place_type.replace(keyword, "").strip()
        
        # 映射中文地點類型到 Google Places API 的類型
        place_type_mapping = {
            "餐廳": "restaurant",
            "咖啡廳": "cafe",
            "咖啡": "cafe",
            "咖啡店": "cafe",
            "會議室": "establishment",
            "會議": "establishment",
            "酒店": "lodging",
            "飯店": "lodging",
            "購物": "shopping_mall",
            "商場": "shopping_mall",
            "超市": "supermarket",
            "銀行": "bank",
            "醫院": "hospital",
            "學校": "school",
            "公園": "park"
        }
        
        # 嘗試找到對應的類型
        api_place_type = None
        for key, value in place_type_mapping.items():
            if key in place_type:
                api_place_type = value
                break
        
        # 使用 Places API 的 text_search 方法搜索附近地點
        # 構建搜索查詢
        search_query = f"{place_type} near {user_location}"
        
        try:
            # 使用 places 方法進行文本搜索
            places_result = gmaps.places(
                query=search_query,
                language='zh-TW'
            )
            
            if places_result.get('results'):
                suggestions = []
                for place in places_result['results'][:max_results]:
                    suggestions.append({
                        "name": place.get('name', ''),
                        "address": place.get('formatted_address', ''),
                        "place_id": place.get('place_id', ''),
                        "rating": place.get('rating', 'N/A')
                    })
                
                return {
                    "success": False,  # 標記為失敗，因為需要用戶選擇
                    "error": f"地址「{query}」太模糊，無法確定確切位置。",
                    "is_ambiguous": True,
                    "suggestions": suggestions,
                    "user_location": user_location
                }
            else:
                # 如果文本搜索失敗，嘗試使用更簡單的查詢
                # 直接使用地點類型名稱搜索
                simple_query = f"{place_type} {user_location}"
                try:
                    simple_result = gmaps.places(
                        query=simple_query,
                        language='zh-TW'
                    )
                    
                    if simple_result.get('results'):
                        suggestions = []
                        for place in simple_result['results'][:max_results]:
                            suggestions.append({
                                "name": place.get('name', ''),
                                "address": place.get('formatted_address', ''),
                                "place_id": place.get('place_id', ''),
                                "rating": place.get('rating', 'N/A')
                            })
                        
                        return {
                            "success": False,
                            "error": f"地址「{query}」太模糊，無法確定確切位置。",
                            "is_ambiguous": True,
                            "suggestions": suggestions,
                            "user_location": user_location
                        }
                except Exception:
                    pass
                
                return {
                    "success": False,
                    "error": f"地址「{query}」太模糊，無法找到確切位置。請提供具體地址或地標名稱。",
                    "is_ambiguous": True
                }
        except Exception as api_error:
            # 如果 Places API 調用失敗，返回友好的錯誤訊息
            return {
                "success": False,
                "error": f"地址「{query}」太模糊，無法找到確切位置。請提供具體地址或地標名稱。\n（API 錯誤：{str(api_error)}）",
                "is_ambiguous": True
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"搜索附近地點失敗：{str(e)}。請提供具體地址或地標名稱。",
            "is_ambiguous": True
        }


def calculate_travel_time(
    origin: str,
    destination: str,
    departure_time: Optional[datetime] = None,
    mode: str = "driving"
) -> Dict[str, any]:
    """
    計算從起點到終點的交通時間
    
    Args:
        origin: 起點地址
        destination: 終點地址
        departure_time: 出發時間（可選，預設為現在）
        mode: 交通方式（'driving', 'walking', 'transit', 'bicycling'），預設為 'driving'
    
    Returns:
        包含以下欄位的字典：
        - success: bool - 是否成功
        - duration_text: str - 交通時間文字描述（例如："25 分鐘"）
        - duration_seconds: int - 交通時間（秒）
        - distance_text: str - 距離文字描述（例如："15.2 公里"）
        - distance_meters: int - 距離（公尺）
        - origin_address: str - 標準化的起點地址
        - destination_address: str - 標準化的終點地址
        - error: str - 錯誤訊息（如果失敗）
    """
    try:
        gmaps = get_gmaps_client()
        
        # 如果沒有指定出發時間，使用現在
        if departure_time is None:
            departure_time = datetime.now()
        
        # 計算路線
        directions_result = gmaps.directions(
            origin=origin,
            destination=destination,
            mode=mode,
            departure_time=departure_time,
            language='zh-TW'  # 使用繁體中文
        )
        
        if not directions_result:
            return {
                "success": False,
                "error": f"無法計算從 {origin} 到 {destination} 的路線"
            }
        
        # 解析結果
        route = directions_result[0]
        leg = route['legs'][0]
        
        duration_text = leg['duration']['text']
        duration_seconds = leg['duration']['value']
        distance_text = leg['distance']['text']
        distance_meters = leg['distance']['value']
        origin_address = leg['start_address']
        destination_address = leg['end_address']
        
        return {
            "success": True,
            "duration_text": duration_text,
            "duration_seconds": duration_seconds,
            "distance_text": distance_text,
            "distance_meters": distance_meters,
            "origin_address": origin_address,
            "destination_address": destination_address
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"計算交通時間失敗：{str(e)}"
        }


def get_user_default_location() -> Optional[str]:
    """
    獲取用戶的預設位置（家或辦公室）
    優先使用辦公室地址，如果沒有則使用家庭地址
    
    Returns:
        預設位置地址字符串，如果都沒有設置則返回 None
    """
    if USER_OFFICE_ADDRESS and USER_OFFICE_ADDRESS.strip():
        return USER_OFFICE_ADDRESS.strip()
    elif USER_HOME_ADDRESS and USER_HOME_ADDRESS.strip():
        return USER_HOME_ADDRESS.strip()
    else:
        return None


def enrich_location_info(location: str, event_datetime: Optional[datetime] = None) -> Dict[str, any]:
    """
    豐富地點資訊：驗證地址、計算交通時間、提供建議
    
    Args:
        location: 地點地址
        event_datetime: 事件時間（用於計算交通時間，可選）
    
    Returns:
        包含豐富資訊的字典：
        - validated: bool - 地址是否有效
        - standardized_address: str - 標準化地址
        - travel_time_info: dict - 交通時間資訊（如果有預設位置）
        - suggestion: str - 建議訊息
    """
    result = {
        "validated": False,
        "standardized_address": location,
        "travel_time_info": None,
        "suggestion": ""
    }
    
    # 1. 驗證並標準化地址
    validation_result = validate_and_standardize_address(location, search_nearby=True)
    
    if not validation_result["success"]:
        error_msg = validation_result.get('error', '地址驗證失敗')
        
        # 如果有建議地點，顯示建議
        if validation_result.get("suggestions"):
            suggestions = validation_result["suggestions"]
            user_location = validation_result.get("user_location", "您的預設位置")
            
            suggestion_text = f"⚠️ {error_msg}\n\n💡 **基於「{user_location}」附近的建議地點：**\n"
            for i, place in enumerate(suggestions, 1):
                rating = place.get('rating', 'N/A')
                suggestion_text += f"{i}. **{place['name']}** - {place['address']}"
                if rating != 'N/A':
                    suggestion_text += f" (評分: {rating}⭐)"
                suggestion_text += "\n"
            suggestion_text += "\n💡 請在事件地點欄位中輸入具體的地點名稱或地址。"
            
            result["suggestion"] = suggestion_text
        else:
            result["suggestion"] = f"⚠️ {error_msg}\n\n💡 **建議：**請提供更具體的地點資訊，例如：\n- 具體地址（如：台北市信義區信義路五段7號）\n- 地標名稱（如：台北101、台北車站）\n- 餐廳/商店名稱（如：星巴克信義店）"
        
        return result
    
    result["validated"] = True
    result["standardized_address"] = validation_result["standardized_address"]
    
    # 檢查是否為模糊地址（需要用戶確認）
    if validation_result.get("is_ambiguous", False):
        result["validated"] = False  # 標記為未完全驗證
        result["suggestion"] = (
            f"⚠️ 地址「{location}」較為模糊，已找到可能的位置：{validation_result['standardized_address']}\n"
            f"💡 建議：請確認這是否為正確地點，或提供更具體的地點資訊。"
        )
        return result
    
    # 2. 如果有預設位置，計算交通時間
    user_location = get_user_default_location()
    if user_location:
        # 如果提供了事件時間，使用事件時間計算；否則使用現在時間
        departure_time = event_datetime if event_datetime else None
        
        travel_result = calculate_travel_time(
            origin=user_location,
            destination=result["standardized_address"],
            departure_time=departure_time
        )
        
        if travel_result["success"]:
            result["travel_time_info"] = travel_result
            
            # 生成建議訊息
            duration = travel_result["duration_text"]
            distance = travel_result["distance_text"]
            
            if event_datetime:
                # 計算建議出發時間（提前 10 分鐘到達）
                suggested_departure = event_datetime - timedelta(
                    seconds=travel_result["duration_seconds"] + 600  # 交通時間 + 10分鐘緩衝
                )
                result["suggestion"] = (
                    f"✅ 地址已驗證：{result['standardized_address']}\n"
                    f"📍 從您的預設位置出發，預計需要 {duration}（{distance}）\n"
                    f"⏰ 建議出發時間：{suggested_departure.strftime('%Y-%m-%d %H:%M')}"
                )
            else:
                result["suggestion"] = (
                    f"✅ 地址已驗證：{result['standardized_address']}\n"
                    f"📍 從您的預設位置出發，預計需要 {duration}（{distance}）"
                )
        else:
            result["suggestion"] = (
                f"✅ 地址已驗證：{result['standardized_address']}\n"
                f"⚠️ 無法計算交通時間：{travel_result.get('error', '未知錯誤')}"
            )
    else:
        result["suggestion"] = (
            f"✅ 地址已驗證：{result['standardized_address']}\n"
            f"💡 提示：在 .env 文件中設置 USER_HOME_ADDRESS 或 USER_OFFICE_ADDRESS 可啟用交通時間計算功能"
        )
    
    return result

