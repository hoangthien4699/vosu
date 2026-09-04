"""Câu thử cho STT. Chọn theo đúng cảnh dùng: họp/thương lượng công việc,
và cố tình nhét từ dễ nghe nhầm."""

# (mã ngôn ngữ, giọng, văn bản)
CASES = [
    # --- tiếng Anh, đối phương nói ---
    ("en", "Samantha", "I think we should table this discussion for now."),
    ("en", "Samantha", "Could you walk me through the pricing structure again?"),
    ("en", "Samantha", "Is there a major blocker we need to resolve first?"),
    ("en", "Samantha", "We can accept the invoice if you can defer the deposit."),
    ("en", "Samantha", "Their team has already sent the revised contract."),
    ("en", "Samantha", "We need a firm commitment on the delivery date."),
    # từ dễ lẫn: accept/except, affect/effect, quarter/quarterly, forty/fourteen
    ("en", "Samantha", "We accept every term except the penalty clause."),
    ("en", "Samantha", "The delay will affect the overall effect on revenue."),
    ("en", "Samantha", "Forty units this quarter, fourteen the next."),
    ("en", "Samantha", "Please confirm whether the license is perpetual or annual."),
    # giọng khác để không chỉ đo trên một chất giọng
    ("en", "Daniel", "I would rather postpone the launch than ship it broken."),
    ("en", "Daniel", "Can you clarify what falls outside the scope of work?"),
    ("en", "Karen", "The budget was approved but the headcount was not."),
    ("en", "Karen", "Let us circle back on that after the legal review."),
    # --- tiếng Việt, người dùng nói ---
    ("vi", "Linh", "Tôi cần thêm thời gian để xem lại hợp đồng."),
    ("vi", "Linh", "Bên anh có thể giảm giá thêm được không?"),
    ("vi", "Linh", "Chúng tôi đồng ý với điều khoản thanh toán."),
    ("vi", "Linh", "Cho tôi hỏi lại về thời gian bàn giao dự án."),
    ("vi", "Linh", "Tôi nghĩ là chưa nên chốt phương án này."),
    ("vi", "Linh", "Anh gửi lại bản báo giá mới nhất giúp tôi nhé."),
]
