import "core-js/modules/es.array.push.js";
import "core-js/modules/es.iterator.constructor.js";
import "core-js/modules/es.iterator.for-each.js";
import * as local from '@/libs/local';
//import HelloWorld from '@/components/HelloWorld.vue';
import { showFailToast, showSuccessToast, showToast } from 'vant';
import Vue from 'vue';
import { Swipe, SwipeItem } from 'vant';
import { Tabbar, TabbarItem } from 'vant';
import { Grid, GridItem } from 'vant';
import * as api from '@/api';
import { showDialog } from 'vant';
import { ref } from 'vue';
import { Tab, Tabs } from 'vant';
import { Col, Row } from 'vant';
import { Popup } from 'vant';
import { Cell, CellGroup } from 'vant';
import { Dialog } from 'vant';
import { NoticeBar } from 'vant';
import { NavBar } from 'vant';
import { Picker } from 'vant';
import { Empty } from 'vant';
import { BackTop } from 'vant';
import { Notify } from 'vant';
import VueClipboard from 'vue-clipboard2';
import 'vant/es/toast/style';
import 'vant/es/dialog/style';
import { ActionSheet } from 'vant';
import { showNotify, closeNotify } from 'vant';
import { DropdownMenu, DropdownItem } from 'vant';
// 引入英文语言包
import 'vant/es/notify/style';
import Header from '../lyout/header.vue';
import Footer from '../lyout/footer.vue';
export default {
  name: 'Home',
  components: {
    Header,
    Footer
  },
  data() {
    return {
      search_value: '',
      lang_list: [],
      select_lang: {},
      lang: {},
      cate: [],
      user: {},
      banner: [],
      goods: [],
      news: [],
      user_name: "",
      pass: "",
      r_user_name: "",
      r_pass: "",
      r_zfpass: "",
      code: "",
      active: 0,
      message: "",
      load: false,
      show: false,
      zhu: "",
      sms: "",
      columns: [{
        text: '+84',
        value: "+84"
      }, {
        text: '+81',
        value: "+81"
      }, {
        text: '+233',
        value: "+233"
      }, {
        text: '+244',
        value: "+244"
      }, {
        text: '+1',
        value: "+1"
      }, {
        text: '+44',
        value: "+44"
      }, {
        text: '+251',
        value: "+251"
      }, {
        text: '+852',
        value: "+852"
      }, {
        text: '+996',
        value: "+996"
      }, {
        text: '+20',
        value: "+20"
      }, {
        text: '+60',
        value: "+60"
      }, {
        text: '+212',
        value: "+212"
      }, {
        text: '+234',
        value: "+234"
      }, {
        text: '+27',
        value: "+27"
      }, {
        text: '+254',
        value: "+254"
      }, {
        text: '+33',
        value: "+33"
      }, {
        text: '+49',
        value: "+49"
      }, {
        text: '+351',
        value: "+351"
      }, {
        text: '+55',
        value: "+55"
      }],
      showPicker: false,
      value: '+84',
      miao: 300,
      yzm_tp: null,
      yzm: {},
      app_root: api.base(),
      yzm_bs: "",
      yzm_code: "",
      yzm_img: "",
      yzm_arr: [{
        key: 0,
        che: 0
      }, {
        key: 1,
        che: 0
      }, {
        key: 2,
        che: 0
      }, {
        key: 3,
        che: 0
      }, {
        key: 4,
        che: 0
      }, {
        key: 5,
        che: 0
      }, {
        key: 6,
        che: 0
      }, {
        key: 7,
        che: 0
      }, {
        key: 8,
        che: 0
      }, {
        key: 9,
        che: 0
      }],
      sliderVisible: false,
      sliderLoading: false,
      sliderOptions: {}
    };
  },
  created: function () {
    let _this = this;
    _this.get_imdex_data();
    this.code = this.$route.query.code ? this.$route.query.code : "0000";
    //  _this.open()
  },
  methods: {
    // 验证
    check(key, value, done, error) {
      let _this = this;
      this.loading = true;
      this.sliderLoading = true;
      api.all('/api/user/send_sms', {
        huo: this.value,
        phone: this.r_user_name,
        yzm: "",
        captcha_key: "",
        yzm_code: this.yzm_code,
        yzm_bs: this.yzm_bs,
        key: key,
        value: value
      }, (err, data) => {
        _this.sliderLoading = false;
        _this.sliderVisible = false;
        if (!err && data.code === 1) {
          showDialog({
            message: data.msg,
            confirmButtonText: this.lang.t11
          }).then(() => {
            // on close
          });
          let send_time = setInterval(() => {
            _this.miao = _this.miao - 1;
            _this.lang.t14 = _this.miao + "S";
            if (_this.miao <= 0) {
              clearInterval(send_time);
              _this.lang.t14 = _this.lang.t15;
              _this.miao = 300;
            }
          }, 1000);
          if (res.pass) {
            done();
          } else {
            error();
          }
        } else {
          showFailToast(data.msg);
        }
      });
    },
    // 关闭触发
    close() {},
    // 获取滑块验证参数
    getSliderOptions() {
      this.sliderLoading = true;
      let _this = this;
      api.all('/api/user/yzm', {
        huo: this.value,
        phone: this.r_user_name
      }, (err, data) => {
        console.log(err);
        if (!err) {
          _this.sliderOptions = {
            sliderImg: data.img,
            sliderKey: data.key,
            sliderY: data.y
          };
          _this.sliderLoading = false;
          _this.sliderVisible = true;
        }
        // }
      });
    },
    che_y(index) {
      this.yzm_arr[index].che = !this.yzm_arr[index].che;
    },
    not_send() {
      this.yzm_tp = null;
      this.yzm = {};
    },
    send() {
      if (this.r_user_name == "") {
        showFailToast(this.lang.t16);
        return;
      }
      if (this.lang.t14 != this.lang.t15) {
        return;
      }
      //let _this=this
      // api.all('/api/user/yzm', { huo: this.value, phone: this.r_user_name }, (err, data) => {
      //   if (!err && data.code === 0) {
      //     showFailToast(data.msg)

      //   }
      //   else {
      //     _this.yzm_arr = [{ key: 0, che: 0 }, { key: 1, che: 0 }, { key: 2, che: 0 }, { key: 3, che: 0 }, { key: 4, che: 0 }, { key: 5, che: 0 }, { key: 6, che: 0 }, { key: 7, che: 0 }, { key: 8, che: 0 }, { key: 9, che: 0 }]
      //   _this.yzm_tp = data.image
      //   _this.yzm = data

      //   }
      // });

      this.getSliderOptions();
    },
    yes_send() {
      let str = "";
      let arr = this.yzm_arr;
      arr.forEach((element, index) => {
        if (element.che) {
          str = str + index.toString();
        }
      });
      this.yzm_tp = null;
      api.all('/api/user/send_sms', {
        huo: this.value,
        phone: this.r_user_name,
        yzm: str,
        captcha_key: this.yzm.captcha_key,
        yzm_code: this.yzm_code,
        yzm_bs: this.yzm_bs
      }, (err, data) => {
        if (!err && data.code === 1) {
          //showSuccessToast(data.msg)

          showDialog({
            message: data.msg,
            confirmButtonText: this.lang.t11
          }).then(() => {
            // on close
          });
          let send_time = setInterval(() => {
            _this.miao = _this.miao - 1;
            _this.lang.t14 = _this.miao + "S";
            if (_this.miao <= 0) {
              clearInterval(send_time);
              _this.lang.t14 = _this.lang.t15;
            }
          }, 1000);
        } else {
          //console.log(data);
          showFailToast(data.msg);
        }
      });
    },
    onConfirm(value) {
      console.log(value);
      this.value = value.selectedValues[0];
      this.showPicker = false;
    },
    onClickLeft() {
      this.lang.chat_off = true;
    },
    //切换语言
    select(e) {
      local.saveInfo('setting_lang', e);
      window.location.reload();
    },
    get_imdex_data: function () {
      api.all('/api/user/init', {}, (err, data) => {
        if (!err && data.code === 1) {
          this.lang_list = data.lang_list;
          this.select_lang = data.select_lang;
          this.lang = data.data.lang;
          this.cate = data.data.cate;
        } else {
          console.log(data);
        }
      });
    },
    login: function (id1) {
      let _this = this;
      _this.$router.push({
        name: 'login',
        query: {
          id: id1
        }
      });
    },
    // 复制成功时的回调函数
    copyText() {
      //console.log(e)
      this.$copyText(this.zhu);
      showNotify({
        type: 'success',
        message: this.lang.ap9
      });
    },
    reg: function (type) {
      api.all('/api/user/reg', {
        user_name: this.r_user_name,
        password: this.r_pass,
        zf_pass: this.r_zfpass,
        code: this.code,
        guo: this.value,
        yzm: this.sms
      }, (err, data) => {
        //showToast(data.msg)
        if (!err && data.code === 1) {
          console.log('login_ok');
          this.show = true;
          this.zhu = data.data.zhu;
          showNotify({
            type: 'success',
            message: data.msg
          });
        } else {
          showNotify(data.msg);
        }
      });
    },
    chat: function (type = 0) {
      this.lang.chat_off = true;
    }
  }
};